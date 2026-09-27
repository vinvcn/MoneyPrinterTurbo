import os
import tempfile
import time
import unittest
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import patch

from app.controllers.manager.base_manager import (
    EXECUTION_MODE_PROCESS,
    EXECUTION_MODE_THREAD,
    resolve_execution_mode,
)
from app.controllers.manager.memory_manager import InMemoryTaskManager
from app.controllers.manager.process_executor import (
    ProcessTaskExecutor,
    run_in_worker,
)


def _add(left: int, right: int, delta: int = 0) -> int:
    """在 worker 进程内执行的普通模块级函数。"""
    return left + right + delta


def _current_pid() -> int:
    """返回执行进程的 pid，用于断言任务确实跨进程执行。"""
    return os.getpid()


def _explode(message: str) -> None:
    raise ValueError(message)


def _write_marker(marker_path: str) -> None:
    """把执行进程的 pid 写进文件，供父进程断言。"""
    with open(marker_path, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


class _RecordingPool:
    """替身执行器：记录 submit 调用并返回已完成的 Future。"""

    def __init__(self, broken: bool = False):
        self.broken = broken
        self.submissions = []
        self.shutdown_calls = []

    def submit(self, func, *args, **kwargs):
        if self.broken:
            raise BrokenProcessPool("worker died")
        self.submissions.append((func, args, kwargs))
        future = Future()
        future.set_result(None)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})


class TestRunInWorker(unittest.TestCase):
    def test_invokes_function_with_args_and_kwargs(self):
        """run_in_worker 必须原样转发位置参数和关键字参数并返回结果。"""
        self.assertEqual(run_in_worker(_add, (1, 2), {"delta": 3}), 6)

    def test_propagates_worker_exception(self):
        """worker 内的异常不能被吞掉，否则调用方会误判任务成功。"""
        with self.assertRaisesRegex(ValueError, "boom"):
            run_in_worker(_explode, ("boom",), {})


class TestProcessTaskExecutor(unittest.TestCase):
    def test_submit_returns_result_from_worker_process(self):
        """提交的任务必须真的跑在另一个进程里，这是 R1 的核心断言。"""
        executor = ProcessTaskExecutor(max_workers=1)
        self.addCleanup(executor.shutdown)

        worker_pid = executor.submit(_current_pid, (), {}).result(timeout=60)

        self.assertIsInstance(worker_pid, int)
        self.assertNotEqual(worker_pid, os.getpid())

    def test_submit_surfaces_exception_raised_in_worker(self):
        """worker 抛出的异常应通过 Future 传回父进程，而不是静默丢失。"""
        executor = ProcessTaskExecutor(max_workers=1)
        self.addCleanup(executor.shutdown)

        future = executor.submit(_explode, ("from worker",), {})

        with self.assertRaisesRegex(ValueError, "from worker"):
            future.result(timeout=60)

    def test_submit_rebuilds_pool_after_broken_pool(self):
        """池损坏（worker 崩溃）时应重建一次并重试，不能永久卡死调度。"""
        broken_pool = _RecordingPool(broken=True)
        healthy_pool = _RecordingPool()
        executor = ProcessTaskExecutor(max_workers=2)

        with patch.object(
            executor,
            "_create_executor",
            side_effect=[broken_pool, healthy_pool],
        ) as create_executor:
            future = executor.submit(_add, (1, 2), {})

        self.assertIsNone(future.result())
        self.assertEqual(create_executor.call_count, 2)
        self.assertEqual(len(healthy_pool.submissions), 1)
        # 损坏的池必须被丢弃，否则它会继续承接任务并再次失败。
        self.assertEqual(len(broken_pool.shutdown_calls), 1)

    def test_submit_propagates_when_rebuilt_pool_is_also_broken(self):
        """重建后仍然损坏必须把异常抛给调用方，由调用方回滚并发名额。"""
        first_pool = _RecordingPool(broken=True)
        second_pool = _RecordingPool(broken=True)
        executor = ProcessTaskExecutor(max_workers=1)

        with patch.object(
            executor,
            "_create_executor",
            side_effect=[first_pool, second_pool],
        ):
            with self.assertRaises(BrokenProcessPool):
                executor.submit(_add, (1, 2), {})

    def test_submit_without_pool_creates_one_lazily(self):
        """池必须懒创建：进程模式但还没有任务时不应启动任何子进程。"""
        executor = ProcessTaskExecutor(max_workers=3)

        self.assertIsNone(executor.executor)

        executor.shutdown()

    def test_shutdown_is_idempotent_and_allows_reuse(self):
        """shutdown 可重复调用，且之后仍能重新提交任务（进程退出路径复用）。"""
        executor = ProcessTaskExecutor(max_workers=1)

        executor.shutdown()
        executor.shutdown()

        self.assertIsNone(executor.executor)
        worker_pid = executor.submit(_current_pid, (), {}).result(timeout=60)
        self.assertNotEqual(worker_pid, os.getpid())
        executor.shutdown()

    def test_max_workers_is_at_least_one(self):
        """max_concurrent_tasks=0 时池仍需合法的 max_workers，不能抛 ValueError。"""
        executor = ProcessTaskExecutor(max_workers=0)
        self.addCleanup(executor.shutdown)

        self.assertEqual(executor.max_workers, 1)


class TestProcessModeTaskManager(unittest.TestCase):
    def test_add_task_executes_in_worker_and_releases_slot(self):
        """
        进程模式下任务必须在子进程里执行，并且执行完成后并发名额要归零；
        否则队列会在一次成功任务后永久阻塞。
        """
        manager = InMemoryTaskManager(
            max_concurrent_tasks=1,
            max_queued_tasks=1,
            execution_mode=EXECUTION_MODE_PROCESS,
        )
        self.addCleanup(manager.shutdown)

        with tempfile.TemporaryDirectory() as tmp_dir:
            marker_path = os.path.join(tmp_dir, "marker.txt")
            manager.add_task(_write_marker, marker_path)

            deadline = time.monotonic() + 60
            while manager.current_tasks and time.monotonic() < deadline:
                time.sleep(0.05)

            self.assertEqual(manager.current_tasks, 0)
            with open(marker_path, encoding="utf-8") as handle:
                worker_pid = int(handle.read().strip())

        self.assertNotEqual(worker_pid, os.getpid())

    def test_add_task_rolls_back_slot_when_submit_fails(self):
        """提交失败必须回滚并发名额，与线程启动失败的行为保持一致。"""
        manager = InMemoryTaskManager(
            max_concurrent_tasks=1,
            max_queued_tasks=1,
            execution_mode=EXECUTION_MODE_PROCESS,
        )
        self.addCleanup(manager.shutdown)

        with patch.object(manager, "_process_executor") as executor:
            executor.submit.side_effect = RuntimeError("pool unavailable")
            with self.assertRaisesRegex(RuntimeError, "pool unavailable"):
                manager.add_task(_add, 1, 2)

        self.assertEqual(manager.current_tasks, 0)

    def test_thread_mode_does_not_create_executor(self):
        """默认线程模式不应创建任何进程执行器，保持原有行为零成本。"""
        manager = InMemoryTaskManager(max_concurrent_tasks=1)

        self.assertEqual(manager.execution_mode, EXECUTION_MODE_THREAD)
        self.assertIsNone(manager._process_executor)


class TestResolveExecutionMode(unittest.TestCase):
    """task_execution_mode 解析必须只返回受支持的模式。"""

    def test_resolves_supported_modes(self):
        cases = [
            # (配置值, redis_enabled, 期望结果)
            ("process", True, EXECUTION_MODE_PROCESS),
            ("thread", True, EXECUTION_MODE_THREAD),
            ("PROCESS", True, EXECUTION_MODE_PROCESS),
            ("  thread  ", True, EXECUTION_MODE_THREAD),
            (None, True, EXECUTION_MODE_THREAD),
            ("", True, EXECUTION_MODE_THREAD),
        ]
        for configured, redis_enabled, expected in cases:
            with self.subTest(configured=configured, redis_enabled=redis_enabled):
                self.assertEqual(
                    resolve_execution_mode(configured, redis_enabled), expected
                )

    def test_falls_back_to_threads_without_redis(self):
        """
        进程模式依赖 Redis 共享任务状态。Redis 关闭时 MemoryState 是进程私有的，
        worker 写入的进度和终态父进程永远看不到，必须降级回线程模式。
        """
        with patch("app.controllers.manager.base_manager.logger") as logger:
            mode = resolve_execution_mode(EXECUTION_MODE_PROCESS, redis_enabled=False)

        self.assertEqual(mode, EXECUTION_MODE_THREAD)
        self.assertTrue(logger.error.called)

    def test_rejects_unknown_mode(self):
        """未知取值应告警并回退到默认线程模式，而不是让服务启动失败。"""
        with patch("app.controllers.manager.base_manager.logger") as logger:
            mode = resolve_execution_mode("goroutine", redis_enabled=True)

        self.assertEqual(mode, EXECUTION_MODE_THREAD)
        self.assertTrue(logger.warning.called)


if __name__ == "__main__":
    unittest.main()
