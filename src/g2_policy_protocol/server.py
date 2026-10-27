"""Shared WebSocket boundary. Backends own model loading and observation conversion.

Requires the optional ``tools`` dependencies; importing the wire contract does not.
"""

import asyncio
import http
import logging
import time
from typing import Protocol

from . import codec
from . import protocol

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

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        from websockets.asyncio.server import serve

        async with serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        from websockets.exceptions import ConnectionClosed

        encoding = codec.encoding_from_path(getattr(getattr(websocket, "request", None), "path", "/"))
        policy = self._policy.new_session() if hasattr(self._policy, "new_session") else self._policy
        await websocket.send(codec.dumps(self._metadata, encoding=encoding))
        while True:
            payload = None
            try:
                raw = await websocket.recv()
                start = time.monotonic()
                payload = codec.loads(raw, encoding=encoding)
                unpack_ms = (time.monotonic() - start) * 1000
                if isinstance(payload, dict) and payload.get("type") == "reset":
                    expected = protocol.build_reset(request_id=payload.get("request_id"), metadata=self._metadata)
                    if payload != expected:
                        raise ValueError("invalid reset message")
                    policy.reset()
                    response = protocol.v3.reset_message(payload["request_id"], ack=True)
                else:
                    request = protocol.parse_request(payload, metadata=self._metadata)
                    infer_start = time.monotonic()
                    result = policy.infer(request)
                    response = protocol.build_response(
                        request_id=request["request_id"],
                        actions=result["actions"],
                        metadata=self._metadata,
                        server_timing={
                            "infer_ms": (time.monotonic() - infer_start) * 1000,
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
