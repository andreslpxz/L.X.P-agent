from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from typing import Literal

IntentAction = Literal["read", "query", "calculate", "write", "delete", "network"]


@dataclass(frozen=True)
class IntentProof:
    action: IntentAction
    target: str
    digest: str
    signature: str

    def to_dict(self) -> dict[str, str]:
        return {
            "action": self.action,
            "target": self.target,
            "digest": self.digest,
            "signature": self.signature,
        }


class CryptographicIntentVerifier:
    def __init__(
        self,
        *,
        allowed_actions: set[IntentAction] | None = None,
        signing_key: bytes | None = None,
    ) -> None:
        self.allowed_actions = allowed_actions or {"read", "query", "calculate"}
        self.signing_key = signing_key or self._load_signing_key()

    def issue(self, action: IntentAction, target: str, rationale: str) -> IntentProof:
        self._ensure_allowed(action)
        digest = self._digest(action, target, rationale)
        signature = hmac.new(self.signing_key, digest.encode(), hashlib.sha256).hexdigest()
        return IntentProof(action=action, target=target, digest=digest, signature=signature)

    def verify(self, proof: IntentProof, rationale: str) -> bool:
        if proof.action not in self.allowed_actions:
            return False
        expected_digest = self._digest(proof.action, proof.target, rationale)
        expected_signature = hmac.new(
            self.signing_key, expected_digest.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(proof.digest, expected_digest) and hmac.compare_digest(
            proof.signature, expected_signature
        )

    def _ensure_allowed(self, action: IntentAction) -> None:
        if action not in self.allowed_actions:
            raise PermissionError(f"Intent action is not pre-approved: {action}")

    @staticmethod
    def _digest(action: IntentAction, target: str, rationale: str) -> str:
        payload = json.dumps(
            {"action": action, "target": target, "rationale": rationale},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _load_signing_key() -> bytes:
        raw_key = os.environ.get("LXP_INTENT_SIGNING_KEY")
        if raw_key:
            return raw_key.encode()
        return secrets.token_bytes(32)
