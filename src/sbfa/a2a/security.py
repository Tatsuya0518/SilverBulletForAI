"""AgentCard signature verification using JWS (RFC 7515) and JCS (RFC 8785).

Prevents AgentCard spoofing even within internal networks.
"""

from __future__ import annotations

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
        AgentCardSignature with the full JWT token and card hash.
    """
    card_data = card.to_json()
    card_data.pop("signature", None)
    canonical = _canonicalize_json(card_data)
    card_hash = hashlib.sha256(canonical).hexdigest()

    # Store the full JWT compact serialization for reliable verification
    token = jwt.encode(
        {"card_hash": card_hash},
        private_key,
        algorithm=algorithm,
    )

    return AgentCardSignature(
        protected=token,
        signature=card_hash,
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

    try:
        payload = jwt.decode(card.signature.protected, public_key, algorithms=[algorithm])
        return payload.get("card_hash") == expected_hash
    except jwt.InvalidTokenError:
        logger.warning("AgentCard '%s' signature verification failed", card.name)
        return False
