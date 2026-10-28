import logging
import os
from typing import Any, Dict, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("auth_manager")


class AuthManager:
    """
    Manages independent authentication sessions and tokens for Software 1, 2, and 3.
    Stores tokens in-memory.
    Provides automatic invalidation and single-retry re-authentication on 401 Unauthorized.
    """

    def __init__(self):
        # In-memory dictionary storing current token per software key ('app1', 'app2', 'app3')
        self.AUTH_TOKENS: Dict[str, str] = {}

    def get_app_config(self, app_key: str) -> Dict[str, str]:
        """Fetch credentials and endpoints for a specific target application."""
        prefix = app_key.upper()
        return {
            "key": app_key,
            "name": os.getenv(f"{prefix}_NAME", f"Software {app_key[-1]}"),
            "url": os.getenv(f"{prefix}_URL", f"http://127.0.0.1:8080/{app_key}").rstrip("/"),
            "username": os.getenv(f"{prefix}_USERNAME", f"admin_{app_key}"),
            "password": os.getenv(f"{prefix}_PASSWORD", f"secret_{app_key}_pass"),
        }

    def invalidate_token(self, app_key: str) -> None:
        """Evict the cached token for a given software."""
        removed = self.AUTH_TOKENS.pop(app_key, None)
        if removed:
            logger.info(f"Invalidated token in memory for [{app_key}]")

    def clear_all_tokens(self) -> None:
        """Clear all active tokens."""
        self.AUTH_TOKENS.clear()

    async def login(self, app_key: str, client: httpx.AsyncClient) -> str:
        """
        Authenticate against the software's login endpoint and cache the session token.
        """
        cfg = self.get_app_config(app_key)
        login_url = f"{cfg['url']}/auth/login"
        payload = {
            "username": cfg["username"],
            "password": cfg["password"],
        }

        logger.info(f"Authenticating [{app_key}] at {login_url}")
        resp = await client.post(login_url, json=payload, timeout=10.0)
        resp.raise_for_status()

        data = resp.json()
        token = data.get("token") or data.get("access_token")
        if not token:
            raise ValueError(f"No token returned by {login_url}: {data}")

        self.AUTH_TOKENS[app_key] = str(token)
        logger.info(f"Successfully obtained session token for [{app_key}]")
        return self.AUTH_TOKENS[app_key]

    async def get_token(self, app_key: str, client: httpx.AsyncClient) -> str:
        """
        Retrieve existing cached token or initiate a login if none exists.
        """
        if app_key in self.AUTH_TOKENS and self.AUTH_TOKENS[app_key]:
            return self.AUTH_TOKENS[app_key]
        return await self.login(app_key, client)

    async def fetch_with_auth_retry(
        self,
        app_key: str,
        path: str,
        client: httpx.AsyncClient,
    ) -> httpx.Response:
        """
        Execute an authenticated GET request against target application.
        If a 401 Unauthorized status is returned, invalidate cached token, re-login once,
        and retry seamlessly.
        """
        cfg = self.get_app_config(app_key)
        base_url = cfg["url"].rstrip("/")
        if path.startswith(f"/{app_key}/"):
            path = path[len(f"/{app_key}"):]
        full_url = f"{base_url}{path}"

        token = await self.get_token(app_key, client)
        headers = {"Authorization": f"Bearer {token}"}

        resp = await client.get(full_url, headers=headers, timeout=10.0)

        # Seamless 401 handling
        if resp.status_code == 401:
            logger.warning(f"Received 401 Unauthorized from [{app_key}] at {full_url}. Auto-invalidating token and retrying login...")
            self.invalidate_token(app_key)
            new_token = await self.login(app_key, client)
            retry_headers = {"Authorization": f"Bearer {new_token}"}
            resp = await client.get(full_url, headers=retry_headers, timeout=10.0)

        resp.raise_for_status()
        return resp


auth_manager = AuthManager()
