"""The shared model runs serially without blocking network/session handling."""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from g2_policy_protocol import codec, protocol
from g2_policy_protocol.server import PolicyServer


def test_inference_keeps_loop_responsive_and_serializes_sessions_and_reset():
    pytest.importorskip("websockets")
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

    class Connection:
        request = SimpleNamespace(path="/")

        def __init__(self):
            self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()

        async def send(self, payload):
            await self.outgoing.put(codec.loads(payload))

        async def recv(self):
            return await self.incoming.get()

    async def run():
        server = PolicyServer(Backend(), metadata=metadata)
        first, second = Connection(), Connection()
        tasks = [asyncio.create_task(server._handler(first))]
        try:
            await first.outgoing.get()  # handshake
            await first.incoming.put(codec.dumps(request))
            assert await asyncio.to_thread(started.wait, 2)
            tasks.append(asyncio.create_task(server._handler(second)))
            assert (await asyncio.wait_for(second.outgoing.get(), 2))["type"] == "metadata"
            assert not finished.is_set()
            await second.incoming.put(codec.dumps(request))
            await asyncio.sleep(0)
            assert calls == [(0, "infer")]
            release.set()
            for connection in (first, second):
                response = await asyncio.wait_for(connection.outgoing.get(), 2)
                protocol.parse_response(response, request_id=1, metadata=metadata)
            await second.incoming.put(codec.dumps(protocol.build_reset(request_id=2, metadata=metadata)))
            assert (await asyncio.wait_for(second.outgoing.get(), 2))["type"] == "reset_ack"
            assert calls == [(0, "infer"), (1, "infer"), (1, "reset")]
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            server._worker.shutdown(wait=True)

    asyncio.run(run())
