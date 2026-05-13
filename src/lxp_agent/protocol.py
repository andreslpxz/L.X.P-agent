from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lxp_agent.intent import CryptographicIntentVerifier, IntentProof

MAX_RESOURCE_BYTES = 24_000
MAX_PROMPT_CHARS = 7_000


@dataclass(frozen=True)
class LatentResource:
    uri: str
    kind: str
    title: str
    digest: str
    summary: str
    embedding: tuple[float, ...]

    def score(self, query_embedding: tuple[float, ...]) -> float:
        return cosine_similarity(self.embedding, query_embedding)


@dataclass(frozen=True)
class Capability:
    name: str
    kind: str
    description: str
    intent: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class GhostResult:
    name: str
    content: str
    proof: IntentProof

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["proof"] = self.proof.to_dict()
        return payload


@dataclass(frozen=True)
class LatentExchangeState:
    query: str
    resources: tuple[LatentResource, ...] = ()
    capabilities: tuple[Capability, ...] = ()
    ghost_results: tuple[GhostResult, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        payload = {
            "protocol": "L.X.P.",
            "mode": "Zero-Hop Context",
            "metadata": self.metadata,
            "semantic_memory": [
                {
                    "uri": resource.uri,
                    "kind": resource.kind,
                    "title": resource.title,
                    "digest": resource.digest,
                    "summary": resource.summary,
                }
                for resource in self.resources
            ],
            "hyper_schema": [
                {
                    "name": capability.name,
                    "kind": capability.kind,
                    "description": capability.description,
                    "intent": capability.intent,
                }
                for capability in self.capabilities
            ],
            "ghost_workers": [result.to_dict() for result in self.ghost_results],
        }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        return f"<LXP_STATE>\n{rendered[:MAX_PROMPT_CHARS]}\n</LXP_STATE>"


class VectorizedMemoryBridge:
    def map_paths(self, scan_paths: Sequence[str]) -> tuple[LatentResource, ...]:
        resources: list[LatentResource] = []
        for raw_path in scan_paths:
            path = Path(raw_path).expanduser().resolve()
            if path.is_file():
                resources.append(self._map_file(path))
            elif path.is_dir():
                for file_path in sorted(path.rglob("*")):
                    if self._is_mappable_file(file_path):
                        resources.append(self._map_file(file_path))
        return tuple(resources)

    def rank(self, query: str, resources: Iterable[LatentResource], *, limit: int = 8) -> tuple[
        LatentResource,
        ...
    ]:
        query_embedding = embed_text(query)
        ranked = sorted(resources, key=lambda resource: resource.score(query_embedding), reverse=True)
        return tuple(ranked[:limit])

    def _map_file(self, path: Path) -> LatentResource:
        content = path.read_bytes()[:MAX_RESOURCE_BYTES]
        text = content.decode("utf-8", errors="replace")
        summary = summarize_text(text)
        return LatentResource(
            uri=path.as_uri(),
            kind="file",
            title=path.name,
            digest=hashlib.sha256(content).hexdigest(),
            summary=summary,
            embedding=embed_text(f"{path.name}\n{text}"),
        )

    @staticmethod
    def _is_mappable_file(path: Path) -> bool:
        if not path.is_file() or path.name.startswith("."):
            return False
        if path.stat().st_size > MAX_RESOURCE_BYTES:
            return False
        return path.suffix.lower() in {
            "",
            ".md",
            ".txt",
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
        }


class HyperSchemaDiscovery:
    def discover(
        self,
        *,
        scan_paths: Sequence[str],
        urls: Sequence[str],
    ) -> tuple[Capability, ...]:
        capabilities: list[Capability] = [
            self._capability(
                "latent.calculate",
                "ghost_worker",
                "Evalúa expresiones matemáticas seguras en paralelo a la inferencia.",
                "calculate",
            )
        ]
        for raw_path in scan_paths:
            path = Path(raw_path).expanduser().resolve()
            if path.exists():
                capabilities.append(
                    self._capability(
                        f"zhc.{path.name}",
                        "zero_hop_context",
                        f"Mapea {path} como memoria semántica vectorizada.",
                        "read",
                    )
                )
        for url in urls:
            capabilities.append(
                self._capability(
                    self._url_capability_name(url),
                    "url_resource",
                    f"Registra {url} como recurso semántico consultable.",
                    "query",
                )
            )
        return tuple(capabilities)

    @staticmethod
    def _capability(name: str, kind: str, description: str, intent: str) -> Capability:
        return Capability(
            name=name,
            kind=kind,
            description=description,
            intent=intent,
            embedding=embed_text(f"{name} {kind} {description} {intent}"),
        )

    @staticmethod
    def _url_capability_name(url: str) -> str:
        digest = hashlib.sha256(url.encode()).hexdigest()[:8]
        return f"semantic.url.{digest}"


class GhostWorkerPool:
    def __init__(self, verifier: CryptographicIntentVerifier | None = None) -> None:
        self.verifier = verifier or CryptographicIntentVerifier()

    def anticipate(self, query: str, urls: Sequence[str]) -> tuple[GhostResult, ...]:
        jobs: list[tuple[str, str]] = []
        expression = extract_math_expression(query)
        if expression:
            jobs.append(("calculate", expression))
        for url in urls[:3]:
            jobs.append(("url_head", url))
        if not jobs:
            return ()

        with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as executor:
            futures = [executor.submit(self._run_job, kind, payload) for kind, payload in jobs]
            return tuple(result for result in (future.result() for future in futures) if result)

    def _run_job(self, kind: str, payload: str) -> GhostResult | None:
        if kind == "calculate":
            rationale = f"Anticipar cálculo matemático para: {payload}"
            proof = self.verifier.issue("calculate", payload, rationale)
            return GhostResult(
                name="latent.calculate",
                content=str(safe_eval_math(payload)),
                proof=proof,
            )
        if kind == "url_head":
            rationale = f"Inspeccionar metadatos públicos de URL: {payload}"
            proof = self.verifier.issue("query", payload, rationale)
            return GhostResult(
                name="semantic.url_probe",
                content=probe_url(payload),
                proof=proof,
            )
        return None


class LXP:
    def __init__(
        self,
        *,
        memory_bridge: VectorizedMemoryBridge | None = None,
        discovery: HyperSchemaDiscovery | None = None,
        ghost_workers: GhostWorkerPool | None = None,
    ) -> None:
        self.memory_bridge = memory_bridge or VectorizedMemoryBridge()
        self.discovery = discovery or HyperSchemaDiscovery()
        self.ghost_workers = ghost_workers or GhostWorkerPool()

    def prepare(
        self,
        query: str,
        *,
        scan_paths: Sequence[str] = (),
        urls: Sequence[str] = (),
    ) -> LatentExchangeState:
        resources = self.memory_bridge.rank(query, self.memory_bridge.map_paths(scan_paths))
        capabilities = self.discovery.discover(scan_paths=scan_paths, urls=urls)
        ghost_results = self.ghost_workers.anticipate(query, urls)
        return LatentExchangeState(
            query=query,
            resources=resources,
            capabilities=capabilities,
            ghost_results=ghost_results,
            metadata={
                "architecture": "semantic-state-not-imperative-tool-calls",
                "security": "hmac-intent-proof-envelope",
            },
        )


def embed_text(text: str, *, dimensions: int = 64) -> tuple[float, ...]:
    vector = [0.0] * dimensions
    for token, count in Counter(tokenize(text)).items():
        index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % dimensions
        vector[index] += 1.0 + math.log(count)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ0-9_./:-]+", text.lower())


def summarize_text(text: str, *, max_chars: int = 700) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = "\n".join(lines[:12])
    if len(summary) > max_chars:
        return summary[: max_chars - 1] + "…"
    return summary


def extract_math_expression(query: str) -> str | None:
    match = re.search(r"(?:calcula|calculate|math|cuánto es|what is)\s*[:=]?\s*([0-9+\-*/(). %^]+)", query, re.I)
    if not match:
        return None
    expression = match.group(1).strip().replace("^", "**")
    return expression or None


def safe_eval_math(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    return _eval_math_node(tree.body, expression)


def _eval_math_node(node: ast.AST, expression: str) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_math_node(node.left, expression)
        right = _eval_math_node(node.right, expression)
        operators = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
            ast.Pow: lambda: left**right,
        }
        for operator_type, operation in operators.items():
            if isinstance(node.op, operator_type):
                return operation()
    if isinstance(node, ast.UnaryOp):
        operand = _eval_math_node(node.operand, expression)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    raise ValueError(f"Unsupported mathematical expression: {expression}")


def probe_url(url: str) -> str:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "lxp-agent/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            return f"{response.status} {response.reason}; content-type={response.headers.get('content-type', '')}"
    except Exception as exc:  # noqa: BLE001
        return f"url_probe_failed: {exc.__class__.__name__}"
