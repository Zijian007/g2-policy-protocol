import asyncio
from types import SimpleNamespace

from g2_policy_protocol import codec


class FakeConnection:
    """In-memory stand-in for a websockets server connection."""

    def __init__(self, encoding="msgpack"):
        self.encoding = encoding
        self.request = SimpleNamespace(path=f"/?encoding={encoding}")
        self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()

    async def send(self, frame):
        await self.outgoing.put(codec.loads(frame, encoding=self.encoding))

    async def recv(self):
        return await self.incoming.get()

    async def put(self, message):
        await self.incoming.put(codec.dumps(message, encoding=self.encoding))

    async def get(self, timeout=2):
        return await asyncio.wait_for(self.outgoing.get(), timeout)
