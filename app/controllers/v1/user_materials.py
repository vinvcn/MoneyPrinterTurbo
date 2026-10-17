"""user_materials v1 路由：Go dispatch 同步镜像接口（plan todo 3）。

契约（todo 9 逐字对接）：
- PUT  /api/v1/user_materials/{owner_id}/{material_id}/files/{idx}?kind=clip|thumb&rev=N
       raw body 流式落盘，1 GiB/文件上限 -> 413；非法 id/kind/idx/rev -> 422
- POST /api/v1/user_materials/{owner_id}/{material_id}/complete  {rev, footages:[...]}
       manifest/文件不齐 -> 409
- GET  /api/v1/user_materials?owner_id=   -> {"materials": [...]}（仅 ready）
- DELETE /api/v1/user_materials/{owner_id}/{material_id} -> 204 幂等

鉴权：无 —— 与既有 POST /video_materials 同一内部信任模型（plan）。
绝不经由 HTTP 返回文件字节（Metis #10）；premise:// 只是逻辑身份。
所有 endpoint 为同步 def（FastAPI 移入线程池执行），sqlite 写经模块锁，
请求体经 anyio 线程->事件循环回跳流式写盘，任何时刻不把 body 整体读进内存。
"""

from __future__ import annotations

import os

import anyio.from_thread
import anyio.to_thread
from fastapi import Query, Request, Response

from app.controllers.v1.base import new_router
from app.models import schema
from app.models.exception import HttpException
from app.services import user_materials

router = new_router()

# 单文件上限 1 GiB（plan；与 VBL 上传配额同数）。测试注入更小值验证 413。
MAX_PUSH_BYTES = 1024 * 1024 * 1024

_TRAVERSAL_HINT = (
    "material path must match /user_materials/{owner_id}/{material_id}/... with "
    "ids ^[A-Za-z0-9_-]{1,64}$ (no embedded slashes)"
)


class _Overflow(Exception):
    def __init__(self, written: int) -> None:
        super().__init__(f"streamed {written} bytes")
        self.written: int = written


def _http(status_code: int, message: str) -> HttpException:
    return HttpException("", status_code=status_code, message=message)


def _parse_int(raw: str, name: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise _http(422, f"invalid {name}: {raw!r}") from None


def _validate_ids(owner_id: str, material_id: str) -> None:
    # url() 构建器同时校验 owner/material_id 字符集；形状错误必须先于业务
    # 语义（如 complete 的 409）以 422 报出。
    try:
        user_materials.url(owner_id, material_id, 0)
    except ValueError as exc:
        raise _http(422, str(exc)) from exc


def _stream_body_to_file(request: Request, dest: str, cap: int) -> int:
    async def _pump() -> int:
        written = 0
        with open(dest, "wb") as sink:
            async for chunk in request.stream():
                written += len(chunk)
                if written > cap:
                    raise _Overflow(written)
                await anyio.to_thread.run_sync(sink.write, chunk)
        return written

    try:
        return anyio.from_thread.run(_pump)
    except _Overflow as exc:
        os.remove(dest)
        raise _http(413, f"file exceeds cap of {cap} bytes (got {exc.written}+)") from exc
    except OSError as exc:
        if os.path.exists(dest):
            os.remove(dest)
        raise _http(500, f"failed to store {os.path.basename(dest)}") from exc


@router.put(
    "/user_materials/{owner_id}/{material_id}/files/{idx}",
    response_model=schema.UserMaterialFilePushResponse,
)
def push_user_material_file(
    request: Request,
    owner_id: str,
    material_id: str,
    idx: str,
    kind: str = Query(default="", description="clip | thumb"),
    rev: str = Query(default="", description="push revision"),
) -> schema.UserMaterialFilePushResponse:
    idx_int = _parse_int(idx, "idx")
    rev_int = _parse_int(rev, "rev")
    try:
        dest = user_materials.put_file(owner_id, material_id, rev_int, idx_int, kind)
    except ValueError as exc:
        raise _http(422, str(exc)) from exc
    written = _stream_body_to_file(request, dest, MAX_PUSH_BYTES)
    return schema.UserMaterialFilePushResponse(
        idx=idx_int, kind=kind, rev=rev_int, bytes_written=written
    )


@router.post(
    "/user_materials/{owner_id}/{material_id}/complete",
    response_model=schema.UserMaterialCompleteResponse,
)
def complete_user_material(
    body: schema.UserMaterialCompleteRequest, owner_id: str, material_id: str
) -> schema.UserMaterialCompleteResponse:
    _validate_ids(owner_id, material_id)
    manifest = [footage.model_dump() for footage in body.footages]
    try:
        result = user_materials.complete(owner_id, material_id, body.rev, manifest)
    except ValueError as exc:
        raise _http(409, str(exc)) from exc
    return schema.UserMaterialCompleteResponse(**result)


@router.get("/user_materials", response_model=schema.UserMaterialListResponse)
def list_user_materials(
    owner_id: str = Query(default="", description="registry tenancy key"),
) -> schema.UserMaterialListResponse:
    try:
        materials = user_materials.list_ready(owner_id)
    except ValueError as exc:
        raise _http(422, str(exc)) from exc
    entries = [schema.UserMaterialEntry(**item) for item in materials]
    return schema.UserMaterialListResponse(materials=entries)


@router.delete("/user_materials/{owner_id}/{material_id}", status_code=204)
def delete_user_material(owner_id: str, material_id: str) -> Response:
    _validate_ids(owner_id, material_id)
    user_materials.delete(owner_id, material_id)
    return Response(status_code=204)


# `%2F` 在路由前已被解码成 "/"，带斜杠的注入会变成"段数正确"以外的形状而
# 掉到 404；下面三个兜底路由把这类路径按契约统一报 422（QA: `..%2Fx` -> 422）。
@router.put("/user_materials/{owner_id}/{tail:path}", include_in_schema=False)
def push_user_material_file_fallback(owner_id: str, tail: str):
    raise _http(422, _TRAVERSAL_HINT)


@router.post("/user_materials/{owner_id}/{tail:path}", include_in_schema=False)
def complete_user_material_fallback(owner_id: str, tail: str):
    raise _http(422, _TRAVERSAL_HINT)


@router.delete("/user_materials/{owner_id}/{tail:path}", include_in_schema=False)
def delete_user_material_fallback(owner_id: str, tail: str):
    raise _http(422, _TRAVERSAL_HINT)
