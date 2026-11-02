"""The shared model runs serially without blocking network/session handling."""

import asyncio
import threading

import numpy as np
from conftest import FakeConnection
from g2_policy_protocol import protocol
from g2_policy_protocol.server import PolicyServer


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
