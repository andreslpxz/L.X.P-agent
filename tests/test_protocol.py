from __future__ import annotations

from pathlib import Path

import pytest

from lxp_agent.intent import CryptographicIntentVerifier, IntentProof
from lxp_agent.protocol import LXP, safe_eval_math


def test_lxp_maps_files_into_zero_hop_context(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("LXP permite memoria semántica y ghost workers.", encoding="utf-8")

    state = LXP().prepare("¿Qué es memoria semántica?", scan_paths=[str(tmp_path)])

    assert state.resources
    assert state.resources[0].kind == "file"
    assert "memoria semántica" in state.resources[0].summary
    assert any(capability.intent == "read" for capability in state.capabilities)


def test_ghost_worker_anticipates_math() -> None:
    state = LXP().prepare("calcula: 2 + 3 * 4")

    assert state.ghost_results[0].name == "latent.calculate"
    assert state.ghost_results[0].content == "14"


def test_intent_verifier_rejects_tampered_proof() -> None:
    verifier = CryptographicIntentVerifier(signing_key=b"test-key")
    proof = verifier.issue("calculate", "2 + 2", "Anticipar cálculo matemático para: 2 + 2")
    tampered = IntentProof(
        action=proof.action,
        target="rm -rf /",
        digest=proof.digest,
        signature=proof.signature,
    )

    assert verifier.verify(proof, "Anticipar cálculo matemático para: 2 + 2")
    assert not verifier.verify(tampered, "Anticipar cálculo matemático para: 2 + 2")


def test_safe_eval_rejects_non_math() -> None:
    with pytest.raises(ValueError):
        safe_eval_math("__import__('os').system('echo nope')")
