"""Real `Transport` implementation (voice_agent_client.py's `Transport` Protocol) backed
by the `websockets` library — connects to the actual AssemblyAI Voice Agent API.

Kept separate from `voice_agent_client.py` on purpose: that module is fully covered by
`tests/unit/test_voice_agent_client.py` using `FakeTransport`, with no network. This
module is the one piece of the agent layer that genuinely cannot be verified without a
live `ASSEMBLYAI_API_KEY` — isolating it here means everything else stays testable, and
anyone reading the test suite can see exactly where the untested boundary is.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

import websockets
from websockets.asyncio.client import ClientConnection


@dataclass
class WebsocketTransport:
    """Thin wrapper around `websockets.asyncio.client.ClientConnection` satisfying the
    `Transport` protocol (`async def send(str)`, `async def recv() -> str`). Use as an
    async context manager:

        async with WebsocketTransport.connect(url, api_key) as transport:
            session = VoiceAgentSession(transport=transport, ...)
            ...
    """

    _connection: ClientConnection

    @classmethod
    async def connect(cls, url: str, api_key: str) -> "WebsocketTransport":
        connection = await websockets.connect(
            url, additional_headers={"Authorization": f"Bearer {api_key}"}
        )
        return cls(_connection=connection)

    async def send(self, message: str) -> None:
        await self._connection.send(message)

    async def recv(self) -> str:
        message = await self._connection.recv()
        return message if isinstance(message, str) else message.decode("utf-8")

    async def close(self) -> None:
        await self._connection.close()

    async def __aenter__(self) -> "WebsocketTransport":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
