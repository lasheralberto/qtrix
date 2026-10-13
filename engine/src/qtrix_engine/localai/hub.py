"""Hugging Face Hub: recommended models, search and GGUF file listing."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

HF_BASE = "https://huggingface.co"

# Verified on the Hub (Oct 2026): instruct models with tool-calling chat templates, Q4_K_M quantisation.
RECOMMENDED: list[dict[str, Any]] = [
    {"repo": "Qwen/Qwen3-4B-GGUF", "file": "Qwen3-4B-Q4_K_M.gguf", "title": "Qwen3 4B",
     "size_gb": 2.5, "ram_gb": 6, "note": "Rápido en portátiles; buen uso de herramientas"},
    {"repo": "Qwen/Qwen3-8B-GGUF", "file": "Qwen3-8B-Q4_K_M.gguf", "title": "Qwen3 8B",
     "size_gb": 5.0, "ram_gb": 10, "note": "Recomendado: el mejor equilibrio calidad/velocidad"},
    {"repo": "bartowski/Qwen2.5-7B-Instruct-GGUF", "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
     "title": "Qwen2.5 7B Instruct", "size_gb": 4.7, "ram_gb": 9, "note": "Muy fiable con llamadas a herramientas"},
    {"repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     "title": "Llama 3.1 8B Instruct", "size_gb": 4.9, "ram_gb": 10, "note": "Generalista, buen español"},
    {"repo": "unsloth/gemma-3-4b-it-GGUF", "file": "gemma-3-4b-it-Q4_K_M.gguf", "title": "Gemma 3 4B",
     "size_gb": 2.5, "ram_gb": 6, "note": "Ligero, multilingüe"},
    {"repo": "bartowski/Qwen2.5-3B-Instruct-GGUF", "file": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
     "title": "Qwen2.5 3B Instruct", "size_gb": 1.9, "ram_gb": 5, "note": "Para equipos con poca memoria"},
    {"repo": "bartowski/Mistral-Nemo-Instruct-2407-GGUF", "file": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
     "title": "Mistral Nemo 12B", "size_gb": 7.5, "ram_gb": 14, "note": "Contexto largo"},
    {"repo": "unsloth/Qwen3-14B-GGUF", "file": "Qwen3-14B-Q4_K_M.gguf", "title": "Qwen3 14B",
     "size_gb": 9.0, "ram_gb": 18, "note": "Máxima calidad local si tienes GPU o 32 GB de RAM"},
]

SPLIT = re.compile(r"^(?P<stem>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)
_QUANT = re.compile(r"(?<![A-Za-z0-9])(I?Q\d(?:_[A-Z0-9]+)*|F16|BF16|F32)(?![A-Za-z0-9])", re.IGNORECASE)


@dataclass
class GgufFile:
    """One downloadable model: a single .gguf or a set of split parts."""

    name: str  # first file of the group
    files: list[str]  # repo paths, in part order
    size: int
    quant: str | None
    sha256: dict[str, str | None] = field(default_factory=dict)
    kind: str = "model"  # model | mmproj

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def quant_of(filename: str) -> str | None:
    base = filename.rsplit("/", 1)[-1].removesuffix(".gguf")
    matches = _QUANT.findall(base)
    return matches[-1].upper() if matches else None


def group_gguf(siblings: list[tuple[str, int | None, str | None]]) -> list[GgufFile]:
    """Group repo files (path, size, sha256) into downloadable models, merging split parts."""
    groups: dict[str, GgufFile] = {}
    for path, size, sha in siblings:
        if not path.lower().endswith(".gguf"):
            continue
        match = SPLIT.match(path)
        key = match.group("stem") if match else path
        kind = "mmproj" if "mmproj" in path.lower() else "model"
        g = groups.setdefault(key, GgufFile(name=path, files=[], size=0, quant=quant_of(key), kind=kind))
        g.files.append(path)
        g.size += size or 0
        g.sha256[path] = sha
    for g in groups.values():
        g.files.sort()
        g.name = g.files[0]
    return sorted(groups.values(), key=lambda g: (g.kind != "model", g.size))


def _api(token: str | None):
    from huggingface_hub import HfApi

    return HfApi(token=token or None)


def search(query: str, *, limit: int = 25, token: str | None = None) -> list[dict[str, Any]]:
    models = _api(token).list_models(search=query or None, filter="gguf", sort="downloads", limit=limit)
    return [
        {"repo": m.id, "downloads": m.downloads, "likes": m.likes,
         "updated": m.last_modified.isoformat() if m.last_modified else None}
        for m in models
    ]


def repo_files(repo: str, *, token: str | None = None) -> dict[str, Any]:
    info = _api(token).model_info(repo, files_metadata=True)
    siblings = [(s.rfilename, s.size, s.lfs.sha256 if s.lfs else None) for s in info.siblings]
    license_ = None
    if info.card_data is not None:
        license_ = getattr(info.card_data, "license", None)
    return {
        "repo": repo,
        "gated": bool(info.gated),
        "downloads": info.downloads,
        "license": license_,
        "files": [g.to_dict() for g in group_gguf(siblings)],
    }


def resolve_url(repo: str, path: str, revision: str = "main") -> str:
    from urllib.parse import quote

    return f"{HF_BASE}/{repo}/resolve/{revision}/{quote(path)}"
