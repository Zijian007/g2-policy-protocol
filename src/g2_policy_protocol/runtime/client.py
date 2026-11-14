"""Synchronous G2 client: handshake, one request at a time, validated replies.

Connection attempts may retry within connect_timeout_s. An inference or reset
is sent once: timeout, disconnect or an invalid reply closes that connection.
Callers own reconnection policy, history sampling and robot action scheduling.
cancel_pending() may be called from another thread to interrupt blocking I/O.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Event
from typing import TYPE_CHECKING, Any

from websockets.sync.client import connect

from .. import protocol
from ..types import Metadata, ObservationContext, RGBImage, WireImage
from . import codec
from .images import prepare_images

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)


@dataclass
class InferStats:
    """Last call's timings (seconds locally, milliseconds from the server).

    letterbox_s includes all image preparation, including stretch and JPEG.
    gpu_ms is the server's infer_ms diagnostic, not a GPU-only measurement.
    Existing names are retained for robot-side telemetry consumers.
    """

    pack_s: float = 0.0
    letterbox_s: float = 0.0
    send_s: float = 0.0
    recv_s: float = 0.0
    wire_bytes: int = 0
    src_hw: str = ""
    image_encoding: str = "raw"
    gpu_ms: float | None = None
    policy_ms: float | None = None
    server_process_ms: float | None = None
    server_decode_ms: float | None = None
    rtc_info: dict[str, Any] | None = None

    @classmethod
    def for_images(cls, images: Mapping[str, RGBImage] | None, encoding: str) -> InferStats:
        sizes = []
        for name, image in (images or {}).items():
            if name in protocol.CAMERA_IDS and getattr(image, "ndim", 0) >= 2:
                sizes.append(f"{name}={image.shape[0]}x{image.shape[1]}")
        return cls(src_hw=",".join(sizes), image_encoding=encoding)

    def record_response(self, response: Mapping[str, Any]) -> None:
        server = response.get("server_timing", {})
        self.gpu_ms = server.get("infer_ms")
        self.policy_ms = response.get("policy_timing", {}).get("infer_ms")
        self.server_process_ms = server.get("process_ms", self.gpu_ms)
        self.server_decode_ms = server.get("decode_ms")
        self.rtc_info = response.get("rtc_info")


class PolicyClient:
    """Connect, inspect metadata, then infer/reset sequentially on one connection.

    infer() accepts RGB uint8 current images; they are resized/encoded according
    to metadata. context.image_history is already prepared in the negotiated
    wire format, so callers can cache historical frames and send them once.
    Override _prepare_images only to supply an optimized image implementation.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        *,
        connect_timeout_s: float = 10.0,
        encoding: str = "msgpack",
        policy_hz: float | None = None,
        require_rtc: bool = False,
    ) -> None:
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{int(port)}"
        self._uri = codec.connection_uri(uri, encoding)
        self._encoding = encoding
        self._policy_hz = policy_hz
        self._require_rtc = require_rtc
        self._connect_timeout_s = float(connect_timeout_s)
        if not math.isfinite(self._connect_timeout_s) or self._connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be positive and finite")
        self._ws = None
        self._cancelled = False
        self._metadata: Metadata | None = None
        self._request_id = 0
        self.last_stats = InferStats()

    @property
    def metadata(self) -> Metadata:
        """Last validated handshake snapshot; raises before the first successful connect."""
        if self._metadata is None:
            raise ConnectionError("connect before reading policy metadata")
        return copy.deepcopy(self._metadata)

    @property
    def uri(self) -> str:
        return self._uri

    def connect(self, *, stop_event: Event | None = None) -> Metadata:
        """Read and validate the first message, retrying only connection failures.

        Invalid metadata fails immediately. Each attempt uses the remaining
        timeout budget. A stop event permits bounded polling during connection.
        """
        if self._cancelled:
            raise ConnectionError("policy client cancelled")
        if stop_event is not None and stop_event.is_set():
            self.cancel_pending()
            raise ConnectionError("policy connect cancelled")
        if self._ws is not None:
            return self.metadata
        deadline = time.monotonic() + self._connect_timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._cancelled or (stop_event is not None and stop_event.is_set()):
                self.cancel_pending()
                raise ConnectionError("policy connect cancelled")
            try:
                budget = max(0.001, deadline - time.monotonic())
                if stop_event is not None:
                    budget = min(budget, 0.25)
                self._ws = connect(self._uri, compression=None, max_size=None, open_timeout=budget)
                if self._cancelled:
                    raise ConnectionError("policy client cancelled")
                budget = max(0.001, deadline - time.monotonic())
                if stop_event is not None:
                    budget = min(budget, 0.25)
                metadata = codec.loads(self._ws.recv(timeout=budget), encoding=self._encoding)
                protocol.validate_metadata(metadata, policy_hz=self._policy_hz, require_rtc=self._require_rtc)
                if self._cancelled or (stop_event is not None and stop_event.is_set()):
                    self.cancel_pending()
                    raise ConnectionError("policy connect cancelled")
                self._metadata = copy.deepcopy(metadata)
                logger.info("connected to %s metadata=%s", self._uri, {k: metadata[k] for k in metadata if k != "type"})
                return self.metadata
            except (TypeError, ValueError):
                self._abort_socket()
                raise
            except Exception as exc:
                last_err = exc
                self._abort_socket()
                if self._cancelled or (stop_event is not None and stop_event.is_set()):
                    self.cancel_pending()
                    raise ConnectionError("policy connect cancelled") from exc
                pause = min(0.2, max(0.0, deadline - time.monotonic()))
                if stop_event is not None:
                    stop_event.wait(pause)
                else:
                    time.sleep(pause)
        raise ConnectionError(f"could not connect to {self._uri}: {last_err}")

    def _prepare_images(
        self,
        images: Mapping[str, RGBImage] | None,
        metadata: Metadata,
    ) -> dict[str, WireImage]:
        """Override for an optimized implementation of the same RGB-to-wire contract."""
        return prepare_images(images, metadata)

    def infer(
        self,
        *,
        prompt: str = "",
        state: ArrayLike | None = None,
        images: Mapping[str, RGBImage] | None = None,
        joint_state: ArrayLike | None = None,
        prev_chunk_left_over: ArrayLike | None = None,
        inference_delay: int = 0,
        rtc_enabled: bool = False,
        recv_timeout_s: float | None = 30.0,
        observation_time_ns: int | None = None,
        context: ObservationContext | None = None,
    ) -> NDArray[np.float32]:
        """Return only validated absolute actions; any remote error raises.

        state is measured pose20; joint_state is measured joint16. Only state
        quantities requested by metadata are sent. History and command reference
        sampling belong to the caller. A failed exchange is never replayed.
        """
        if self._cancelled:
            raise ConnectionError("policy client cancelled")
        if self._ws is None:
            self.connect()
        metadata = self._metadata
        assert metadata is not None  # connect() validated the handshake.
        stats = self.last_stats = InferStats.for_images(images, metadata["image_encoding"])
        start = time.monotonic()
        prepared = None
        if context is None or "image_history" not in context:
            prepared = self._prepare_images(images, metadata)
            stats.letterbox_s = time.monotonic() - start
        self._request_id += 1
        spec = metadata["observation_spec"]
        request = protocol.build_request(
            request_id=self._request_id,
            prompt=prompt,
            state=state if spec["pose"] else None,
            joint_state=joint_state if spec["joint"] else None,
            images=prepared,
            metadata=metadata,
            context=context,
            observation_time_ns=observation_time_ns,
            rtc_enabled=rtc_enabled,
            inference_delay=inference_delay if rtc_enabled else 0,
            prev_chunk_left_over=prev_chunk_left_over if rtc_enabled else None,
        )
        stats.pack_s = time.monotonic() - start
        response = self._exchange(request, recv_timeout_s=recv_timeout_s)
        try:
            actions = protocol.parse_response(response, request_id=self._request_id, metadata=metadata)
        except Exception:
            self._abort_socket()
            raise
        stats.record_response(response)
        if self._cancelled:
            raise ConnectionError("policy client cancelled")
        return actions

    def reset(self, *, recv_timeout_s: float | None = 30.0) -> None:
        """Wait until this connection's backend has cleared its episode state."""
        if self._cancelled:
            raise ConnectionError("policy client cancelled")
        if self._ws is None:
            self.connect()
        self._request_id += 1
        metadata = self._metadata
        assert metadata is not None
        request = protocol.build_reset(request_id=self._request_id, metadata=metadata)
        response = self._exchange(request, recv_timeout_s=recv_timeout_s)
        try:
            protocol.parse_reset_ack(response, request_id=self._request_id)
        except Exception:
            self._abort_socket()
            raise

    def _exchange(self, request: Mapping[str, Any], *, recv_timeout_s: float | None) -> Any:
        """Encode, send once, receive once. All exchange failures discard the socket."""
        if self._cancelled:
            raise ConnectionError("policy client cancelled")
        websocket = self._ws
        if websocket is None:
            raise ConnectionError("policy connection closed before sending request")
        try:
            start = time.monotonic()
            packed = codec.dumps(request, encoding=self._encoding)
            self.last_stats.pack_s += time.monotonic() - start
            self.last_stats.wire_bytes = len(packed.encode("utf-8")) if isinstance(packed, str) else len(packed)
            start = time.monotonic()
            websocket.send(packed)
            self.last_stats.send_s = time.monotonic() - start
            start = time.monotonic()
            frame = websocket.recv(timeout=recv_timeout_s)
            response = codec.loads(frame, encoding=self._encoding)
            self.last_stats.recv_s = time.monotonic() - start
            return response
        except Exception as exc:
            self._abort_socket()
            if self._cancelled:
                raise ConnectionError("policy client cancelled") from exc
            if isinstance(exc, TimeoutError):
                raise TimeoutError(f"policy request timed out after {recv_timeout_s}s on {self._uri}") from exc
            raise

    def close(self) -> None:
        websocket, self._ws = self._ws, None
        if websocket is not None:
            try:
                websocket.close()
            except Exception:
                logger.debug("policy connection close failed", exc_info=True)

    def cancel_pending(self) -> None:
        """Interrupt pending I/O and permanently disable this client's reconnects."""
        self._cancelled = True
        self._abort_socket()

    def _abort_socket(self) -> None:
        websocket, self._ws = self._ws, None
        if websocket is not None:
            try:
                websocket.close_socket()
            except OSError:
                logger.debug("socket close failed", exc_info=True)

    def __enter__(self) -> PolicyClient:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()
