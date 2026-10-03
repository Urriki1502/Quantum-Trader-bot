from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class JsonHttpError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, retry_after: str | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class JsonHttpTransport(Protocol):
    async def request_json(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        json_body: Mapping[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> Mapping[str, Any]:
        ...


@dataclass(slots=True)
class UrllibJsonTransport:
    """Small stdlib-only async JSON transport.

    Network I/O is moved to a worker thread so the engine event loop remains
    responsive. No credentials or provider-specific behavior live here.
    """

    user_agent: str = "QuantumTrader/2.0"

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        json_body: Mapping[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> Mapping[str, Any]:
        return await asyncio.to_thread(
            self._request_json_sync,
            method,
            url,
            params=params,
            json_body=json_body,
            timeout=timeout,
        )

    def _request_json_sync(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, str | int] | None,
        json_body: Mapping[str, Any] | None,
        timeout: float,
    ) -> Mapping[str, Any]:
        if params:
            query = urlencode({k: str(v) for k, v in params.items()})
            url = f"{url}{'&' if '?' in url else '?'}{query}"

        body: bytes | None = None
        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if json_body is not None:
            body = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
                status = getattr(response, "status", 200)
        except HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            payload = exc.read().decode("utf-8", errors="replace")
            raise JsonHttpError(
                f"HTTP {exc.code}: {payload[:300]}",
                status=exc.code,
                retry_after=retry_after,
            ) from exc
        except URLError as exc:
            raise JsonHttpError(f"network error: {exc.reason}") from exc

        if status < 200 or status >= 300:
            raise JsonHttpError(f"unexpected HTTP status {status}", status=status)

        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JsonHttpError("response was not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise JsonHttpError("JSON response must be an object")
        return decoded
