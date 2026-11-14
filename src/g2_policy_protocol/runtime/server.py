"""WebSocket sessions around the typed PolicyBackend.infer boundary.

Each connection receives metadata, then exchanges obs/action or reset/reset_ack.
See _handle_request for the complete observation -> inference -> response path.
The runtime decodes images; backends own model loading and model adaptation.
Image decoding and inference are serialized on one worker; networking stays async.
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .. import protocol
from ..types import (
    InferenceResult,
    Metadata,
    Observation,
    ObservationRequest,
)
from . import codec
from .images import ImageDecodeError, decode_observation

logger = logging.getLogger(__name__)


class PolicyBackend(Protocol):
    """Minimum backend interface: decoded robot observation -> absolute targets.

    Stateless backends only need infer(). Images are RGB uint8 arrays; image
    histories are already selected and decoded. Only model adaptation remains.
    Do not build response envelopes or echo request IDs here: the server does it.
    """

    def infer(self, observation: Observation) -> InferenceResult: ...


class ResettablePolicyBackend(PolicyBackend, Protocol):
    """A backend that clears its episode caches when reset is advertised.

    reset() completes before reset_ack is sent. It does not move the robot or
    reload shared model weights. Backends without episode state may use a no-op.
    """

    def reset(self) -> None: ...


class PolicySessionFactory(Protocol):
    """Optional factory creating independent episode state per connection.

    A factory may share model weights across the backends it returns. Each
    returned backend must implement reset() when metadata advertises reset.
    Without a factory, the same backend serves every connection and must not
    mix per-connection episode state. Inference/reset remain globally serialized.
    """

    def new_session(self) -> PolicyBackend: ...


class PolicyServer:
    """Serve one backend: handshake, validation, request IDs, errors and a single model thread."""

    def __init__(
        self,
        policy: PolicyBackend | PolicySessionFactory,
        *,
        metadata: Metadata,
        host: str = "0.0.0.0",
        port: int | None = 8000,
    ) -> None:
        protocol.validate_metadata(metadata)
        self._policy, self._metadata = policy, dict(metadata)
        self._host, self._port = host, port
        if not callable(getattr(policy, "new_session", None)):
            self._validate_backend(policy)
        # A shared model has one execution lane. Wait for the lock before submitting
        # work so concurrent connections cannot build an unbounded executor queue.
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="g2-policy")
        self._model_lock = asyncio.Lock()

    def _validate_backend(self, backend: object) -> None:
        """Validate the actual session, rather than assuming the factory is one."""
        if not callable(getattr(backend, "infer", None)):
            raise ValueError("backend requires infer")
        if self._metadata["reset_supported"] and not callable(getattr(backend, "reset", None)):
            raise ValueError("reset_supported requires backend.reset")

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
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
        encoding = codec.encoding_from_path(websocket.request.path)
        create_session = getattr(self._policy, "new_session", None)
        session = create_session() if callable(create_session) else self._policy
        self._validate_backend(session)
        await websocket.send(codec.dumps(self._metadata, encoding=encoding))
        while True:
            try:
                raw = await websocket.recv()
            except ConnectionClosed:
                return
            response = await self._handle_request(session, raw, encoding)
            try:
                await websocket.send(codec.dumps(response, encoding=encoding))
            except ConnectionClosed:
                return

    async def _handle_request(self, session: PolicyBackend, raw: bytes | str, encoding: str):
        """One exchange: decode/validate -> worker -> validate result -> response.

        Input errors are invalid_request. Model/reset/output errors are
        inference_error. Every response carries the incoming request ID.
        """
        start = time.monotonic()
        payload = None
        try:
            payload = codec.loads(raw, encoding=encoding)
            unpack_ms = (time.monotonic() - start) * 1000
            is_reset = isinstance(payload, dict) and payload.get("type") == "reset"
            request = (
                protocol.parse_reset(payload, metadata=self._metadata)
                if is_reset
                else protocol.parse_request(payload, metadata=self._metadata)
            )
        except Exception as exc:
            return _error_response(payload, "invalid_request", exc)

        try:
            async with self._model_lock:
                loop = asyncio.get_running_loop()
                if is_reset:
                    await loop.run_in_executor(self._worker, session.reset)
                    return protocol.build_reset_ack(request_id=request["request_id"])
                result, timing = await loop.run_in_executor(self._worker, self._infer, session, request)
            return protocol.build_response(
                request_id=request["request_id"],
                actions=result["actions"],
                metadata=self._metadata,
                server_timing={
                    **result.get("server_timing", {}),
                    **timing,
                    "unpack_ms": unpack_ms,
                    "process_ms": (time.monotonic() - start) * 1000,
                },
                policy_timing=result.get("policy_timing"),
                rtc_info=result.get("rtc_info"),
            )
        except ImageDecodeError as exc:
            return _error_response(payload, "invalid_request", exc)
        except Exception as exc:
            return _error_response(payload, "inference_error", exc)

    def _infer(
        self,
        session: PolicyBackend,
        request: ObservationRequest,
    ) -> tuple[InferenceResult, dict[str, float]]:
        """One worker job: decode exactly the frames used by this observation."""
        start = time.monotonic()
        observation = decode_observation(request, self._metadata)
        decode_ms = (time.monotonic() - start) * 1000
        start = time.monotonic()
        result = session.infer(observation)
        return result, {"decode_ms": decode_ms, "infer_ms": (time.monotonic() - start) * 1000}


def _error_response(payload, code: str, error: Exception):
    """Echo a valid ID even when the rest of the request was rejected."""
    logger.warning("G2 request failed: %s", error)
    request_id = payload.get("request_id", 0) if isinstance(payload, dict) else 0
    return protocol.build_error(
        request_id=request_id if type(request_id) is int and request_id >= 0 else 0,
        code=code,
        message=f"{type(error).__name__}: {error}",
    )


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
