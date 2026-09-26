import threading
from typing import Any, Callable, Dict

from loguru import logger

from app.controllers.manager.process_executor import ProcessTaskExecutor
from app.models import const


EXECUTION_MODE_THREAD = "thread"
EXECUTION_MODE_PROCESS = "process"
_SUPPORTED_EXECUTION_MODES = (EXECUTION_MODE_THREAD, EXECUTION_MODE_PROCESS)


def resolve_execution_mode(configured_mode: Any, redis_enabled: bool) -> str:
    """
    解析 ``task_execution_mode``，并在不安全的组合下强制回退到线程模式。

    进程模式下每个 worker 是独立解释器，任务状态必须经由 Redis 共享。
    ``enable_redis = false`` 时使用 MemoryState（进程私有），worker 写入的进度
    和终态父进程永远读不到，任务会在 API 里永远显示"生成中"，因此这种组合
    必须降级而不是硬着头皮跑。
    """
    mode = str(configured_mode or "").strip().lower()
    if not mode:
        return EXECUTION_MODE_THREAD

    if mode not in _SUPPORTED_EXECUTION_MODES:
        logger.warning(
            f"unsupported task_execution_mode: {configured_mode!r}, "
            f"fallback to {EXECUTION_MODE_THREAD}"
        )
        return EXECUTION_MODE_THREAD

    if mode == EXECUTION_MODE_PROCESS and not redis_enabled:
        logger.error(
            "task_execution_mode = process requires enable_redis = true, "
            f"fallback to {EXECUTION_MODE_THREAD}"
        )
        return EXECUTION_MODE_THREAD

    return mode


class TaskQueueFullError(ValueError):
    pass


class TaskManager:
    def __init__(
        self,
        max_concurrent_tasks: int,
        max_queued_tasks: int = 100,
        execution_mode: str = EXECUTION_MODE_THREAD,
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queued_tasks = max_queued_tasks
        self.execution_mode = execution_mode
        self.current_tasks = 0
        # 必须可重入：进程模式下若 Future 在 add_done_callback 注册时已经完成，
        # 回调会在当前线程内联执行，而这个线程正持有 self.lock（add_task /
        # check_queue 都是持锁调用 execute_task）。用非重入的 Lock 会自死锁。
        self.lock = threading.RLock()
        self.queue = self.create_queue()
        # 进程池按需创建：进入进程模式但尚无任务时不应启动任何子进程。
        self._process_executor = (
            ProcessTaskExecutor(max_concurrent_tasks)
            if execution_mode == EXECUTION_MODE_PROCESS
            else None
        )

    def create_queue(self):
        raise NotImplementedError()

    def add_task(self, func: Callable, *args: Any, **kwargs: Any):
        with self.lock:
            if self.current_tasks < self.max_concurrent_tasks:
                logger.info(
                    f"add task: {func.__name__}, current_tasks: {self.current_tasks}"
                )
                # 在线程启动前先预占并发名额。原实现在线程内部递增，连续请求
                # 可能都在子线程获得锁之前看到 current_tasks=0，从而突破并发
                # 上限。启动失败时回滚名额，让后续请求仍可正常调度。
                self.current_tasks += 1
                try:
                    self.execute_task(func, *args, **kwargs)
                except Exception:
                    self.current_tasks -= 1
                    raise
            else:
                queue_size = self.queue_size()
                # 并发数已满时才进入排队。队列必须有上限，否则匿名接口可以持续
                # 堆积任务对象和请求参数，最终造成内存耗尽或第三方 API 成本失控。
                if queue_size >= self.max_queued_tasks:
                    logger.warning(
                        f"reject task: {func.__name__}, queue_size: {queue_size}, "
                        f"max_queued_tasks: {self.max_queued_tasks}"
                    )
                    raise TaskQueueFullError("task queue is full, please try again later")

                logger.info(
                    f"enqueue task: {func.__name__}, current_tasks: {self.current_tasks}, "
                    f"queue_size: {queue_size}"
                )
                self.enqueue({"func": func, "args": args, "kwargs": kwargs})

    def submit_idempotent(
        self,
        state,
        task_id: str,
        params_hash: str,
        owner_token: str,
        task_fields: dict,
        func: Callable,
        task_kwargs: dict,
    ) -> str:
        """Atomically accept idempotent work, then dispatch it from the queue."""
        from app.services.state import IdempotentAcceptance

        task_info = {"func": func, "args": (), "kwargs": task_kwargs}
        with self.lock:
            # 幂等路径先接受入队、后调度（accept -> check_queue），与 add_task
            # 的先调度、后入队时序不同：接受时可供立即派发的槽位（available_slots）
            # 会在随后的 check_queue 中清出队列，因此队列容量按
            # max_queued_tasks + available_slots 计算，接受后队列仍不超过上限。
            available_slots = max(
                0, self.max_concurrent_tasks - self.current_tasks
            )
            queue_capacity = self.max_queued_tasks + available_slots
            try:
                acceptance = IdempotentAcceptance(
                    task_id=task_id,
                    params_hash=params_hash,
                    owner_token=owner_token,
                    task_fields=task_fields,
                    task_info=task_info,
                    queue_capacity=queue_capacity,
                )
                outcome = state.accept_idempotent_task(acceptance, self)
            except Exception:
                state.abort_idempotent_task(task_id, owner_token)
                raise

        if outcome == const.IDEMPOTENCY_QUEUE_FULL:
            state.abort_idempotent_task(task_id, owner_token)
            return outcome
        if outcome != const.IDEMPOTENCY_ACCEPTED:
            return outcome

        try:
            self.check_queue()
        except Exception as exc:
            # Acceptance is already durable. check_queue restores the item when
            # thread creation fails, so the caller may safely receive success.
            logger.exception(
                f"accepted task remains queued after dispatch failure: {task_id}: {exc}"
            )
        return outcome

    def execute_task(self, func: Callable, *args: Any, **kwargs: Any):
        if self.execution_mode != EXECUTION_MODE_PROCESS:
            thread = threading.Thread(
                target=self.run_task, args=(func, *args), kwargs=kwargs
            )
            thread.start()
            return

        # 进程模式：任务函数和参数整体序列化到 worker 进程。self 不参与序列化
        # （它持有 Redis 客户端、队列和锁，都不可 pickle），名额释放改由父进程
        # 的 Future 回调完成。
        # submit 抛出的异常必须原样上抛：add_task / check_queue 依赖它回滚已经
        # 预占的并发名额。
        future = self._process_executor.submit(func, args, kwargs)
        future.add_done_callback(self._on_process_task_finished)

    def _on_process_task_finished(self, future) -> None:
        """
        worker 进程结束后释放并发名额。

        回调在进程池的管理线程里执行，异常不会传给任何调用方，因此这里既要
        记录 worker 侧的失败，也要保证 task_done 一定被调用——否则一次失败就
        会永久占用一个并发额度，队列最终完全停摆。
        """
        try:
            if future.cancelled():
                logger.warning("process task was cancelled before it started")
            else:
                error = future.exception()
                if error is not None:
                    logger.error(
                        f"process task failed: {type(error).__name__}: {error}"
                    )
        except Exception as exc:
            logger.exception(f"failed to inspect finished process task: {exc}")
        finally:
            self.task_done()

    def shutdown(self) -> None:
        """释放进程执行器；线程模式没有需要清理的后台资源。"""
        if self._process_executor is not None:
            self._process_executor.shutdown()

    def run_task(self, func: Callable, *args: Any, **kwargs: Any):
        try:
            func(*args, **kwargs)  # call the function here, passing *args and **kwargs.
        finally:
            self.task_done()

    def check_queue(self):
        with self.lock:
            if (
                self.current_tasks < self.max_concurrent_tasks
                and not self.is_queue_empty()
            ):
                task_info = self.dequeue()
                if task_info is None:
                    # dequeue() may skip and discard queue entries that no longer
                    # pass current validation (see RedisTaskManager.dequeue) and
                    # return None once nothing usable is left, even though
                    # is_queue_empty() was False a moment earlier.
                    return
                func = task_info["func"]
                args = task_info.get("args", ())
                kwargs = task_info.get("kwargs", {})
                # 与直接创建任务保持同一计数时机，避免刚出队的任务尚未在线程
                # 内计数时，又有新请求绕过队列占用同一个并发名额。
                self.current_tasks += 1
                try:
                    self.execute_task(func, *args, **kwargs)
                except Exception:
                    self.current_tasks -= 1
                    self.enqueue(task_info)
                    raise

    def task_done(self):
        with self.lock:
            self.current_tasks -= 1
        self.check_queue()

    def enqueue(self, task: Dict):
        raise NotImplementedError()

    def enqueue_transaction(self, pipeline, task: Dict):
        """Append a job through a backend transaction, or directly for memory."""
        if pipeline is not None:
            raise TypeError("this task manager does not support Redis transactions")
        self.enqueue(task)

    def dequeue(self):
        raise NotImplementedError()

    def is_queue_empty(self):
        raise NotImplementedError()

    def queue_size(self):
        raise NotImplementedError()