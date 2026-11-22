"""Configuration centralisée : chemins du projet et lecture des fichiers YAML de configs/.

Aucun chemin en dur : tout est relatif à la racine du projet (surchargeable
avec la variable d'environnement EURLEX_RAG_ROOT).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# src/eurlex_rag/config.py -> parents[2] = racine du dépôt
PROJECT_ROOT = Path(os.environ.get("EURLEX_RAG_ROOT", Path(__file__).resolve().parents[2]))
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"

CORPUS_CONFIG = CONFIGS_DIR / "corpus.yaml"
CHUNKING_CONFIG = CONFIGS_DIR / "chunking.yaml"
MANIFEST_PATH = RAW_DIR / "manifest.json"

# Code langue court (nos dossiers) -> code ISO 639-2 (API Cellar)
LANG_ISO3 = {"fr": "fra", "en": "eng"}


@dataclass(frozen=True)
class TextSpec:
    key: str
    celex: str
    label: str
    expected_articles: int


@dataclass(frozen=True)
class CorpusConfig:
    languages: tuple[str, ...]
    texts: tuple[TextSpec, ...]


def load_corpus(path: Path = CORPUS_CONFIG) -> CorpusConfig:
    """Lit et valide le fichier corpus.yaml."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    languages = tuple(data["languages"])
    unknown = [lang for lang in languages if lang not in LANG_ISO3]
    if unknown:
        raise ValueError(f"Langues non supportées : {unknown} (supportées : {list(LANG_ISO3)})")

    texts = tuple(TextSpec(**item) for item in data["texts"])
    keys = [t.key for t in texts]
    if len(keys) != len(set(keys)):
        raise ValueError("Clés dupliquées dans corpus.yaml")
    return CorpusConfig(languages=languages, texts=texts)


@dataclass(frozen=True)
class ChunkingConfig:
    tokenizer: str
    single_chunk_max_tokens: int
    target_tokens: int
    max_tokens: int
    min_tokens: int
    parent_max_tokens: int
    definition_titles: tuple[str, ...]
    display_names: dict[str, dict[str, str]]
    labels: dict[str, dict[str, str]]


def load_chunking(path: Path = CHUNKING_CONFIG) -> ChunkingConfig:
    """Lit et valide configs/chunking.yaml."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = ChunkingConfig(
        tokenizer=data["tokenizer"],
        single_chunk_max_tokens=int(data["single_chunk_max_tokens"]),
        target_tokens=int(data["target_tokens"]),
        max_tokens=int(data["max_tokens"]),
        min_tokens=int(data["min_tokens"]),
        parent_max_tokens=int(data["parent_max_tokens"]),
        definition_titles=tuple(t.lower() for t in data.get("definition_titles", [])),
        display_names=data.get("display_names", {}),
        labels=data["labels"],
    )
    if not (cfg.min_tokens < cfg.target_tokens <= cfg.max_tokens):
        raise ValueError("chunking.yaml : il faut min_tokens < target_tokens <= max_tokens")
    if cfg.single_chunk_max_tokens > cfg.max_tokens:
        raise ValueError("chunking.yaml : single_chunk_max_tokens ne doit pas dépasser max_tokens")
    return cfg