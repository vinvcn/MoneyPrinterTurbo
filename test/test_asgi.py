import asyncio
from unittest.mock import patch

from app import asgi
from app.controllers.manager.base_manager import (
    EXECUTION_MODE_PROCESS,
    EXECUTION_MODE_THREAD,
)


def _run_lifespan():
    """进入再退出一次 ASGI 生命周期，触发启动钩子和 finally 里的清理。"""

    async def _cycle():
        async with asgi.application_lifespan(None):
            return None

    return asyncio.run(_cycle())


def test_lifespan_shuts_down_task_manager():
    """退出时必须释放任务执行器，否则 worker 池只能靠解释器 atexit 兜底。"""
    with (
        patch("app.services.task.recover_interrupted_cross_posts"),
        patch("app.controllers.v1.video.task_manager") as task_manager,
    ):
        _run_lifespan()

    task_manager.shutdown.assert_called_once_with()


def test_thread_mode_emits_no_worker_local_warning():
    """默认线程模式没有进程私有状态问题，不应产生任何启动期告警。"""
    with (
        patch("app.services.task.recover_interrupted_cross_posts"),
        patch("app.controllers.v1.video.task_manager"),
        patch(
            "app.controllers.v1.video.task_execution_mode", EXECUTION_MODE_THREAD
        ),
        patch("app.asgi.config") as app_config,
        patch("app.asgi.logger") as asgi_logger,
    ):
        app_config.app.get.side_effect = lambda key, default=None: default
        _run_lifespan()

    asgi_logger.warning.assert_not_called()


def test_process_mode_warns_about_worker_local_state():
    """进程模式必须同时提示 cross-post 注册表和本地 Whisper 模型会按 worker 复制。"""
    values = {"upload_post_auto_upload": True, "subtitle_provider": "whisper"}
    with (
        patch("app.services.task.recover_interrupted_cross_posts"),
        patch("app.controllers.v1.video.task_manager"),
        patch(
            "app.controllers.v1.video.task_execution_mode", EXECUTION_MODE_PROCESS
        ),
        patch("app.asgi.config") as app_config,
        patch("app.asgi.logger") as asgi_logger,
    ):
        app_config.app.get.side_effect = (
            lambda key, default=None: values.get(key, default)
        )
        _run_lifespan()

    assert asgi_logger.warning.call_count == 2


def test_process_mode_without_hazards_stays_quiet():
    """进程模式但两项都没开启时，不应产生任何告警。"""
    with (
        patch("app.services.task.recover_interrupted_cross_posts"),
        patch("app.controllers.v1.video.task_manager"),
        patch(
            "app.controllers.v1.video.task_execution_mode", EXECUTION_MODE_PROCESS
        ),
        patch("app.asgi.config") as app_config,
        patch("app.asgi.logger") as asgi_logger,
    ):
        app_config.app.get.side_effect = lambda key, default=None: default
        _run_lifespan()

    asgi_logger.warning.assert_not_called()
