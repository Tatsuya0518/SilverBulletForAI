"""AgentCard signature verification using JWS (RFC 7515) and JCS (RFC 8785).

Prevents AgentCard spoofing even within internal networks.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging

import jwt

from sbfa.a2a.agent_card import AgentCard, AgentCardSignature

logger = logging.getLogger(__name__)


def _canonicalize_json(data: dict) -> bytes:
    """JSON Canonicalization Scheme (RFC 8785).

    Produces a deterministic JSON representation for signing.
    Uses sorted keys and no whitespace (simplified JCS).
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_agent_card(card: AgentCard, private_key: str, algorithm: str = "RS256") -> AgentCardSignature:
    """Sign an AgentCard using JWS.

    Args:
        card: The AgentCard to sign.
        private_key: PEM-encoded private key.
        algorithm: JWS algorithm (default: RS256).

    Returns:
        AgentCardSignature with protected header and signature.
    """
    card_data = card.to_json()
    card_data.pop("signature", None)
    canonical = _canonicalize_json(card_data)

    header = {"alg": algorithm, "typ": "JWT"}
    protected = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")

    token = jwt.encode(
        {"card_hash": hashlib.sha256(canonical).hexdigest()},
        private_key,
        algorithm=algorithm,
    )
    parts = token.split(".")

    return AgentCardSignature(
        protected=protected,
        signature=parts[2] if len(parts) == 3 else parts[-1],
    )


def verify_agent_card(card: AgentCard, public_key: str, algorithm: str = "RS256") -> bool:
    """Verify an AgentCard's JWS signature.

    Args:
        card: The AgentCard with signature to verify.
        public_key: PEM-encoded public key.
        algorithm: JWS algorithm (default: RS256).

    Returns:
        True if signature is valid, False otherwise.
    """
    if not card.signature:
        logger.warning("AgentCard '%s' has no signature", card.name)
        return False

    card_data = card.to_json()
    card_data.pop("signature", None)
    canonical = _canonicalize_json(card_data)
    expected_hash = hashlib.sha256(canonical).hexdigest()

    token = f"{card.signature.protected}.{base64.urlsafe_b64encode(json.dumps({'card_hash': expected_hash}).encode()).decode().rstrip('=')}.{card.signature.signature}"

    try:
        payload = jwt.decode(token, public_key, algorithms=[algorithm])
        return payload.get("card_hash") == expected_hash
    except jwt.InvalidTokenError:
        logger.warning("AgentCard '%s' signature verification failed", card.name)
        return False
