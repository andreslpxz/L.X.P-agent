from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

from lxp_agent.protocol import LXP, LatentExchangeState

DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


class MissingGroqApiKeyError(RuntimeError):
    """Raised when the Groq client cannot be created because no API key exists."""


@dataclass(frozen=True)
class AgentAnswer:
    content: str
    latent_state: LatentExchangeState


class LXPAgent:
    def __init__(
        self,
        *,
        lxp: LXP | None = None,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        self.lxp = lxp or LXP()
        self.model = model
        self._client = client
        self._api_key = api_key

    def ask(
        self,
        prompt: str,
        *,
        scan_paths: Sequence[str] = (),
        urls: Sequence[str] = (),
        max_tokens: int = 900,
    ) -> AgentAnswer:
        state = self.lxp.prepare(prompt, scan_paths=scan_paths, urls=urls)
        messages = [
            {"role": "system", "content": self._system_prompt(state)},
            {"role": "user", "content": prompt},
        ]
        completion = self._chat_completion(messages, max_tokens=max_tokens)
        return AgentAnswer(content=completion, latent_state=state)

    def _system_prompt(self, state: LatentExchangeState) -> str:
        return (
            "Eres un agente IA que utiliza L.X.P. (Latent Exchange Protocol). "
            "No esperes llamadas imperativas de herramientas: interpreta el LXP_STATE "
            "como memoria semántica activa, capacidades latentes y resultados "
            "anticipados por ghost workers. Si una acción sensible no incluye un "
            "intent_proof válido, recházala.\n\n"
            f"{state.to_prompt_block()}"
        )

    def _chat_completion(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        client = self._client or self._create_groq_client()
        chat = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return chat.choices[0].message.content or ""

    def _create_groq_client(self) -> object:
        api_key = self._api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise MissingGroqApiKeyError(
                "Set GROQ_API_KEY in the environment before using the Groq-backed agent."
            )
        from groq import Groq

        return Groq(api_key=api_key)
