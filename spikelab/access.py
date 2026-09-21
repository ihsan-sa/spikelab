"""Public mode's gate: every request carries a Cloudflare Access assertion this server verifies, or it is refused.

Cloudflare Access signs the visitor in before a packet reaches this machine and adds a signed JWT to each request
in the Cf-Access-Jwt-Assertion header. Nothing here can tell whether Access is really in front of the hostname, so
this check stands on its own and fails closed: no header, a key that cannot be fetched, or any claim that does not
match is a 403.

Checked on every request: the RS256 signature against the team's published keys (cached; an unknown key id makes
PyJWKClient fetch them again), aud (this Access application's AUD tag), iss (https://<team>.cloudflareaccess.com),
exp and nbf, and the email claim on the allow-list, compared case-folded.
"""
from __future__ import annotations

import re

import jwt

HEADER = "Cf-Access-Jwt-Assertion"
TEAM_RE = re.compile(r"[a-z0-9][a-z0-9-]*\Z")  # interpolated into the key URL and the issuer, so a bare name only


class Access:
    def __init__(self, team: str, aud: str, emails: list[str], certs_url: str | None = None):
        if not TEAM_RE.match(team or ""):
            raise ValueError(f"access team must be a bare team name, got {team!r}")
        self.allow = {e.strip().casefold() for e in emails if e.strip()}
        if not aud or not self.allow:
            raise ValueError("public mode needs an access aud and at least one allowed email")
        self.aud, self.issuer = aud, f"https://{team}.cloudflareaccess.com"
        # certs_url is for tests only: a throwaway key served on localhost
        self.keys = jwt.PyJWKClient(certs_url or f"{self.issuer}/cdn-cgi/access/certs", timeout=10)

    def email(self, headers) -> str | None:
        """The verified, allowed email on this request, or None."""
        token = (headers.get(HEADER) or "").strip()
        if not token:
            return None
        try:
            key = self.keys.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["RS256"], audience=self.aud, issuer=self.issuer,
                                options={"require": ["exp", "iss", "aud"]})
        except Exception:  # PyJWT's error family, or the key fetch failing: either way, no
            return None
        email = str(claims.get("email") or "").strip().casefold()
        return email if email in self.allow else None
