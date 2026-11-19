"""Configuration centralisée : chemins du projet et lecture de configs/corpus.yaml.

Aucun chemin en dur : tout est relatif à la racine du projet (surchargeable
avec la variable d'environnement EURLEX_RAG_ROOT).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# src/eurlex_rag/config.py -> parents[2] = racine du dépôt
PROJECT_ROOT = Path(os.environ.get("EURLEX-RAG", Path(__file__).resolve().parents[2]))
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"

CORPUS_CONFIG = CONFIGS_DIR / "corpus.yaml"
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
