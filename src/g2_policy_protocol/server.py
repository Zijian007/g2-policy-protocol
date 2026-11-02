"""Shared WebSocket server. Backends own model loading and observation conversion."""

import asyncio
import http
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from . import _messages as messages
from . import codec, protocol
from .images import ImageDecodeError

logger = logging.getLogger(__name__)


class PolicyBackend(Protocol):
    """One session owns episode state. Neural models may be shared by sessions."""

    def infer(self, request: dict) -> dict: ...
    def reset(self) -> None: ...
    def new_session(self) -> "PolicyBackend": ...


class _RequestError(Exception):
    """Wraps a failure with the error code to report to the client."""

    def __init__(self, code, cause):
        super().__init__(cause)
        self.code, self.cause = code, cause


class PolicyServer:
    """Serve one backend: handshake, validation, request IDs, errors and a single model thread."""

    def __init__(self, policy, *, metadata, host="0.0.0.0", port=8000):
        protocol.validate_metadata(metadata)
        self._policy, self._metadata = policy, dict(metadata)
        self._host, self._port = host, port
        if metadata.get("reset_supported") and not callable(getattr(policy, "reset", None)):
            raise ValueError("reset_supported requires backend.reset")
        # A shared model has one execution lane. Wait for the lock before submitting
        # work so concurrent connections cannot build an unbounded executor queue.
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
        session = self._policy.new_session() if hasattr(self._policy, "new_session") else self._policy
        await websocket.send(codec.dumps(self._metadata, encoding=encoding))
        while True:
            try:
                raw = await websocket.recv()
            except ConnectionClosed:
                return
            payload = None
            try:
                start = time.monotonic()
                try:
                    payload = codec.loads(raw, encoding=encoding)
                except Exception as exc:
                    raise _RequestError("invalid_request", exc) from exc
                unpack_ms = (time.monotonic() - start) * 1000
                if isinstance(payload, dict) and payload.get("type") == "reset":
                    response = await self._handle_reset(session, payload)
                else:
                    response = await self._handle_obs(session, payload, start, unpack_ms)
            except _RequestError as error:
                logger.exception("G2 request failed")
                response = protocol.build_error(
                    request_id=_reply_id(payload),
                    code=error.code,
                    message=f"{type(error.cause).__name__}: {error.cause}",
                )
            try:
                await websocket.send(codec.dumps(response, encoding=encoding))
            except ConnectionClosed:
                return

    async def _run_model(self, function, *args):
        """Run backend work serially, distinguishing invalid images from model failures."""
        try:
            async with self._model_lock:
                return await asyncio.get_running_loop().run_in_executor(self._worker, function, *args)
        except ImageDecodeError as exc:
            raise _RequestError("invalid_request", exc) from exc
        except Exception as exc:
            raise _RequestError("inference_error", exc) from exc

    async def _handle_reset(self, session, payload):
        try:
            messages.check_reset(payload)
            if not self._metadata["reset_supported"]:
                raise ValueError("server does not support episode reset")
        except ValueError as exc:
            raise _RequestError("invalid_request", exc) from exc
        await self._run_model(session.reset)
        return protocol.build_reset_ack(request_id=payload["request_id"])

    async def _handle_obs(self, session, payload, start, unpack_ms):
        try:
            request = protocol.parse_request(payload, metadata=self._metadata)
        except Exception as exc:
            raise _RequestError("invalid_request", exc) from exc
        infer_start = time.monotonic()
        result = await self._run_model(session.infer, request)
        infer_ms = (time.monotonic() - infer_start) * 1000
        try:
            return protocol.build_response(
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
        except Exception as exc:
            raise _RequestError("inference_error", exc) from exc


def _reply_id(payload):
    """Echo a valid request_id; otherwise reply with 0."""
    request_id = payload.get("request_id", 0) if isinstance(payload, dict) else 0
    return request_id if type(request_id) is int and request_id >= 0 else 0


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
