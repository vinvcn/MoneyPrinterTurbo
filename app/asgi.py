"""Application implementation - ASGI."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import config
from app.controllers.manager.base_manager import EXECUTION_MODE_PROCESS
from app.models.exception import HttpException
from app.router import root_api_router
from app.utils import utils


def _warn_worker_local_state() -> None:
    """
    进程模式启动时提示哪些状态是 worker 进程私有的。

    这两项在进程模式下会落到每个 worker 解释器里，父进程（也就是 API）读不到，
    所以启动时明确告警，而不是等线上出现诡异行为再排查。
    """
    from app.controllers.v1 import video as video_controller

    if video_controller.task_execution_mode != EXECUTION_MODE_PROCESS:
        return

    if config.app.get("upload_post_auto_upload", False):
        logger.warning(
            "task_execution_mode = process with upload_post_auto_upload: "
            "cross-post futures live in the worker process, so the API cannot "
            "see in-flight uploads and task deletion is no longer blocked by them"
        )

    if str(config.app.get("subtitle_provider", "edge")).strip().lower() == "whisper":
        logger.warning(
            "task_execution_mode = process with subtitle_provider = whisper: "
            "each worker process loads its own Whisper model"
        )


@asynccontextmanager
async def application_lifespan(_: FastAPI):
    """集中处理 API 进程启动恢复和关闭日志。"""
    logger.info("startup event")
    _warn_worker_local_state()

    # 跨平台发布由当前进程线程池执行，不会在服务重启后恢复。启动时把 Redis
    # 中确认已失去执行进程的活动状态收敛为失败，避免任务永久无法删除。
    from app.services import task as task_service

    task_service.recover_interrupted_cross_posts()
    try:
        yield
    finally:
        # 进程模式下要显式释放 worker 池，避免退出顺序交给 atexit 猜。shutdown
        # 不取消排队任务，已开跑的 worker 会跑完当前任务，与线程模式下非守护
        # 线程在解释器退出前继续执行的行为一致。
        from app.controllers.v1 import video as video_controller

        video_controller.task_manager.shutdown()
        logger.info("shutdown event")


def exception_handler(request: Request, e: HttpException):
    return JSONResponse(
        status_code=e.status_code,
        content=utils.get_response(e.status_code, e.data, e.message),
    )


def validation_exception_handler(request: Request, e: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content=utils.get_response(
            status=400, data=e.errors(), message="field required"
        ),
    )


def get_application() -> FastAPI:
    """Initialize FastAPI application.

    Returns:
       FastAPI: Application object instance.

    """
    instance = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=False,
        lifespan=application_lifespan,
    )
    instance.include_router(root_api_router)
    instance.add_exception_handler(HttpException, exception_handler)
    instance.add_exception_handler(RequestValidationError, validation_exception_handler)
    return instance


app = get_application()

# Configures the CORS middleware for the FastAPI app
cors_allowed_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
origins = cors_allowed_origins_str.split(",") if cors_allowed_origins_str else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

task_dir = utils.task_dir()
app.mount(
    "/tasks", StaticFiles(directory=task_dir, html=True, follow_symlink=True), name=""
)

public_dir = utils.public_dir()
app.mount("/", StaticFiles(directory=public_dir, html=True), name="")