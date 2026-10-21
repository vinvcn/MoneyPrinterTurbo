"""scene_cut v1 路由（BYOM footage KB，ADR-0023 决策 3）。

POST /api/v1/scene_cut {"input_path": "..."} -> {"scenes": [{"t_start_s","t_end_s"}]}

鉴权：无 —— 与 user_materials / video_materials 同一内部信任模型（VBL worker
是唯一调用方，经 docker 网络直达）。绝不经由 HTTP 返回文件字节（ADR-0023 决策 3）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import anyio.to_thread
from fastapi import Request

from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.models.schema import SceneCutRequest, SceneCutResponse
from app.services import scene_cut as scene_cut_service

router = new_router()


@router.post("/scene_cut", response_model=SceneCutResponse)
async def scene_cut(payload: SceneCutRequest, request: Request) -> SceneCutResponse:
    """检测 ``input_path`` 的场景边界（pyscenedetect，CPU 密集 → 线程池执行）。"""
    request_id = base.get_task_id(request)
    input_path = payload.input_path

    # 调用方（VBL worker）把源视频放在共享交换卷 /shared 上再传路径（ADR-0023
    # 决策 3）；路径不存在即 404，与仓库既有 file 端点同一语义。
    if not Path(input_path).is_file():
        raise HttpException(
            task_id=request_id,
            status_code=404,
            message=f"{request_id}: input path is not a file: {input_path}",
        )

    try:
        boundaries = await anyio.to_thread.run_sync(
            scene_cut_service.detect_scene_cuts, input_path
        )
    except scene_cut_service.SceneCutError as exc:
        # 打不开 / 不是视频 / 解码失败都归 422：路径合法但内容不可切分。
        raise HttpException(
            task_id=request_id,
            status_code=422,
            message=f"{request_id}: {exc}",
        ) from exc

    scenes = [boundary.as_dict() for boundary in boundaries]
    logging.getLogger(__name__).info(
        "scene_cut done, request_id=%s, scenes=%d, input=%s",
        request_id,
        len(scenes),
        input_path,
    )
    return SceneCutResponse(scenes=scenes)
