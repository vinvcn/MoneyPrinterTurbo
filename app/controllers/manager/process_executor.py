"""跨进程任务执行器。

线程模式下所有并发任务共享同一个解释器和 GIL，CPU 密集的合成阶段会互相
串行化（见 artifacts/performance-analysis-20260926.md 的 F1）。这个模块把
任务放进独立进程执行，让每个 worker 拥有自己的 GIL。

刻意使用 spawn 而不是 fork：父进程持有 FFmpeg/GPU 相关句柄，fork 会把它们
复制到子进程，容易造成句柄共享和驱动状态错乱。
"""

from __future__ import annotations

import contextlib
import multiprocessing
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Dict, Tuple

from loguru import logger


def run_in_worker(
    func: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """在 worker 进程内调用任务函数。

    `func` 必须是模块级可 pickle 的对象（例如 `app.services.task.start`），
    否则 `ProcessPoolExecutor.submit` 会在父进程同步抛出序列化异常，让调用方
    及时回滚并发名额，而不是让任务静默消失。
    """
    return func(*args, **kwargs)


class ProcessTaskExecutor:
    """按需创建的 spawn 进程池。

    池只在第一次 submit 时创建，因此进程模式在没有任务时不会启动任何子进程。
    worker 异常退出（段错误、OOM）会让池进入 broken 状态：已经提交的任务会通过
    Future 收到 `BrokenProcessPool`，之后的 submit 会同步抛出同一异常，此时
    丢弃旧池并重建一次即可恢复调度。
    """

    def __init__(self, max_workers: int):
        self.max_workers = max(1, int(max_workers))
        self._lock = threading.Lock()
        self._executor: ProcessPoolExecutor | None = None

    @property
    def executor(self) -> ProcessPoolExecutor | None:
        """当前进程池（未创建时为 None），供诊断和测试观察。"""
        with self._lock:
            return self._executor

    def _create_executor(self) -> ProcessPoolExecutor:
        context = multiprocessing.get_context("spawn")
        return ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=context,
        )

    def _get_executor(self) -> ProcessPoolExecutor:
        with self._lock:
            if self._executor is None:
                self._executor = self._create_executor()
            return self._executor

    def _discard_executor(self) -> None:
        """丢弃当前池，不阻塞等待（等待会卡住调度线程）。"""
        with self._lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            # 已经 broken 的池 shutdown 不会抛业务异常；保守起见仍然压制，
            # 因为丢弃失败不应该阻止重建。
            with contextlib.suppress(Exception):
                executor.shutdown(wait=False)

    def submit(
        self,
        func: Callable[..., Any],
        args: Tuple[Any, ...] = (),
        kwargs: Dict[str, Any] | None = None,
    ) -> Future:
        """提交任务并返回 Future；池损坏时重建一次再重试。"""
        payload = (func, tuple(args), dict(kwargs or {}))

        try:
            executor = self._get_executor()
            return executor.submit(run_in_worker, *payload)
        except BrokenProcessPool:
            logger.warning(
                "task worker pool is broken, rebuilding "
                f"(max_workers={self.max_workers})"
            )
            self._discard_executor()
            # 第二次失败必须原样抛出：调用方（TaskManager.add_task /
            # check_queue）依赖这个异常回滚已预占的并发名额并把任务放回队列。
            return self._get_executor().submit(run_in_worker, *payload)

    def shutdown(self) -> None:
        """关闭进程池；可重复调用，之后再次 submit 会重新建池。"""
        self._discard_executor()
