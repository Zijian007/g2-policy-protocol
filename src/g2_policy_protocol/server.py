"""Shared WebSocket boundary. Backends own model loading and observation conversion."""

import asyncio
import http
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from . import codec, protocol

logger = logging.getLogger(__name__)


class PolicyBackend(Protocol):
    """One session owns episode state. Neural models may be shared by sessions."""

    def infer(self, request: dict) -> dict: ...
    def reset(self) -> None: ...
    def new_session(self) -> "PolicyBackend": ...


class PolicyServer:
    def __init__(self, policy, *, metadata, host="0.0.0.0", port=8000):
        protocol.validate_metadata(metadata)
        self._policy, self._metadata = policy, dict(metadata)
        self._host, self._port = host, port
        if metadata.get("reset_supported") and not callable(getattr(policy, "reset", None)):
            raise ValueError("reset_supported requires backend.reset")
        # A shared model has one execution lane. Wait before submitting work so
        # concurrent connections cannot build an unbounded executor queue.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="g2-policy")
        self._model_lock = asyncio.Lock()

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        try:
            async with serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                max_queue=1,
                process_request=_health_check,
            ) as server:
                await server.serve_forever()
        finally:
            self._worker.shutdown(wait=True, cancel_futures=True)

    async def _handler(self, websocket):
        encoding = codec.encoding_from_path(getattr(getattr(websocket, "request", None), "path", "/"))
        loop = asyncio.get_running_loop()
        policy = self._policy.new_session() if hasattr(self._policy, "new_session") else self._policy
        await websocket.send(codec.dumps(self._metadata, encoding=encoding))
        while True:
            payload = None
            error_code = "invalid_request"
            try:
                raw = await websocket.recv()
                start = time.monotonic()
                payload = codec.loads(raw, encoding=encoding)
                unpack_ms = (time.monotonic() - start) * 1000
                if isinstance(payload, dict) and payload.get("type") == "reset":
                    expected = protocol.build_reset(request_id=payload.get("request_id"), metadata=self._metadata)
                    if payload != expected:
                        raise ValueError("invalid reset message")
                    error_code = "inference_error"
                    async with self._model_lock:
                        await loop.run_in_executor(self._worker, policy.reset)
                    response = protocol.build_reset_ack(request_id=payload["request_id"])
                else:
                    request = protocol.parse_request(payload, metadata=self._metadata)
                    error_code = "inference_error"
                    async with self._model_lock:
                        infer_start = time.monotonic()
                        result = await loop.run_in_executor(self._worker, policy.infer, request)
                        infer_ms = (time.monotonic() - infer_start) * 1000
                    response = protocol.build_response(
                        request_id=request["request_id"],
                        actions=result["actions"],
                        metadata=self._metadata,
                        server_timing={
                            "infer_ms": infer_ms,
                            "unpack_ms": unpack_ms,
                            "process_ms": (time.monotonic() - start) * 1000,
                            **result.get("server_timing", {}),
                        },
                        policy_timing=result.get("policy_timing"),
                        rtc_info=result.get("rtc_info"),
                    )
                await websocket.send(codec.dumps(response, encoding=encoding))
            except ConnectionClosed:
                return
            except Exception as exc:
                logger.exception("G2 request failed")
                request_id = payload.get("request_id", 0) if isinstance(payload, dict) else 0
                if type(request_id) is not int or request_id < 0:
                    request_id = 0
                try:
                    await websocket.send(
                        codec.dumps(
                            protocol.build_error(
                                request_id=request_id,
                                code=error_code,
                                message=f"{type(exc).__name__}: {exc}",
                            ),
                            encoding=encoding,
                        )
                    )
                except ConnectionClosed:
                    return


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
