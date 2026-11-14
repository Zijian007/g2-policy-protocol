"""The shared model runs serially without blocking network/session handling."""

import asyncio
import threading

import numpy as np
import pytest
from conftest import FakeConnection

from g2_policy_protocol import protocol
from g2_policy_protocol.runtime.server import PolicyServer


def test_inference_keeps_loop_responsive_and_serializes_sessions_and_reset():
    metadata = protocol.build_metadata(
        policy_hz=30, action_horizon=1, required_cameras=[], reset_supported=True,
    )
    request = protocol.build_request(
        request_id=1, prompt="test", state=np.zeros(20, np.float32), images={}, metadata=metadata,
    )
    started, release, finished = (threading.Event() for _ in range(3))
    calls = []
    sessions = []

    class Backend:
        def new_session(self):
            session = type(self)()
            session.index = len(sessions)
            sessions.append(session)
            return session

        def infer(self, request):
            calls.append((self.index, "infer"))
            if self.index == 0:
                started.set()
                assert release.wait(3), "network loop did not release the first inference"
                finished.set()
            return {"actions": np.zeros((1, 20), np.float32)}

        def reset(self):
            calls.append((self.index, "reset"))

    async def run():
        server = PolicyServer(Backend(), metadata=metadata)
        first, second = FakeConnection(), FakeConnection()
        tasks = [asyncio.create_task(server._handler(first))]
        try:
            await first.get()  # handshake
            await first.put(request)
            assert await asyncio.to_thread(started.wait, 2)
            tasks.append(asyncio.create_task(server._handler(second)))
            assert (await second.get())["type"] == "metadata"
            assert not finished.is_set()
            await second.put(request)
            await asyncio.sleep(0)
            assert calls == [(0, "infer")]
            release.set()
            for connection in (first, second):
                response = await connection.get()
                protocol.parse_response(response, request_id=1, metadata=metadata)
            await second.put(protocol.build_reset(request_id=2, metadata=metadata))
            assert (await second.get())["type"] == "reset_ack"
            assert calls == [(0, "infer"), (1, "infer"), (1, "reset")]
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            server._worker.shutdown(wait=True)

    asyncio.run(run())


def test_factory_only_object_can_create_resettable_sessions():
    metadata = protocol.build_metadata(policy_hz=30, action_horizon=1, required_cameras=[], reset_supported=True)
    created = []

    class Session:
        def __init__(self):
            self.steps = 0

        def infer(self, request):
            self.steps += 1
            return {"actions": np.full((1, 20), self.steps, np.float32)}

        def reset(self):
            self.steps = 0

    class Factory:
        # Shared weights could live here; only sessions own episode state.
        def new_session(self):
            session = Session()
            created.append(session)
            return session

    async def run():
        server = PolicyServer(Factory(), metadata=metadata)
        connections = [FakeConnection(), FakeConnection()]
        handlers = [asyncio.create_task(server._handler(connection)) for connection in connections]
        try:
            for connection in connections:
                await connection.get()
            for connection in (connections[0], connections[0], connections[1]):
                await connection.put(protocol.build_request(
                    request_id=1, prompt="test", state=np.zeros(20), images={}, metadata=metadata,
                ))
                await connection.get()
            assert [session.steps for session in created] == [2, 1]
            await connections[0].put(protocol.build_reset(request_id=2, metadata=metadata))
            ack = await connections[0].get()
            protocol.parse_reset_ack(ack, request_id=2)
            assert [session.steps for session in created] == [0, 1]
        finally:
            for task in handlers:
                task.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)
            server._worker.shutdown(wait=True)

    asyncio.run(run())


def test_reset_capability_is_checked_on_the_actual_session_before_handshake():
    metadata = protocol.build_metadata(policy_hz=30, required_cameras=[], reset_supported=True)

    class InferenceOnly:
        def infer(self, request):
            return {"actions": np.zeros((1, 20), np.float32)}

    class Factory:
        def new_session(self):
            return InferenceOnly()

    async def run():
        server = PolicyServer(Factory(), metadata=metadata)
        connection = FakeConnection()
        try:
            with pytest.raises(ValueError, match="reset_supported requires backend.reset"):
                await server._handler(connection)
            assert connection.outgoing.empty()
        finally:
            server._worker.shutdown(wait=True)

    asyncio.run(run())


def test_image_decode_runs_on_worker_while_other_connections_can_handshake(monkeypatch):
    from g2_policy_protocol.runtime import server as runtime

    metadata = protocol.build_metadata(policy_hz=30, action_horizon=1, required_cameras=[])
    decoding, release = threading.Event(), threading.Event()
    original_decode = runtime.decode_observation
    workers = []

    def decode(request, metadata):
        workers.append(threading.current_thread().name)
        decoding.set()
        assert release.wait(2)
        return original_decode(request, metadata)

    class Backend:
        def infer(self, observation):
            return {"actions": np.zeros((1, 20), np.float32)}

    monkeypatch.setattr(runtime, "decode_observation", decode)

    async def run():
        server = PolicyServer(Backend(), metadata=metadata)
        first, second = FakeConnection(), FakeConnection()
        tasks = [asyncio.create_task(server._handler(first))]
        try:
            await first.get()
            await first.put(protocol.build_request(
                request_id=1, prompt="test", state=np.zeros(20), images={}, metadata=metadata,
            ))
            assert await asyncio.to_thread(decoding.wait, 2)
            tasks.append(asyncio.create_task(server._handler(second)))
            assert (await second.get())["type"] == "metadata"
            release.set()
            response = await first.get()
            protocol.parse_response(response, request_id=1, metadata=metadata)
            assert response["server_timing"]["decode_ms"] >= 0
            assert workers[0].startswith("g2-policy")
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            server._worker.shutdown(wait=True)

    asyncio.run(run())
