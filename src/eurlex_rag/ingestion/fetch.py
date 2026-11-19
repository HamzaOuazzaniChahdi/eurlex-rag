"""Étape 1 — Téléchargement du corpus EUR-Lex (FR/EN) vers data/raw/.

Idempotent : un fichier déjà présent ET référencé dans le manifest n'est pas
re-téléchargé (sauf avec --force). Le manifest est mis à jour après chaque fichier.

Usage :
    uv run eurlex-fetch                    # tout le corpus
    uv run eurlex-fetch --only rgpd --lang fr
    uv run eurlex-fetch --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import requests

from eurlex_rag import config
from eurlex_rag.config import CorpusConfig, TextSpec

log = logging.getLogger(__name__)

MIN_BYTES = 50_000  # en dessous : page de challenge / erreur, pas un texte de loi
RETRIES = 3
BACKOFF_S = 3
DELAY_S = 2  # pause entre deux téléchargements (politesse)
TIMEOUT_S = 60
USER_AGENT = "Mozilla/5.0 (compatible; eurlex-rag research project)"

Downloader = Callable[[TextSpec, str], tuple[bytes, str]]
ManifestKey = tuple[str, str]  # (key, lang)


def _sources(celex: str, lang: str) -> list[tuple[str, dict[str, str]]]:
    """URLs à essayer, par ordre : API Cellar puis page EUR-Lex."""
    return [
        (
            f"http://publications.europa.eu/resource/celex/{celex}",
            {"Accept": "application/xhtml+xml", "Accept-Language": config.LANG_ISO3[lang]},
        ),
        (f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/HTML/?uri=CELEX:{celex}", {}),
    ]


def download(spec: TextSpec, lang: str) -> tuple[bytes, str]:
    """Télécharge un texte dans une langue. Retourne (contenu, url_source)."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    last_error = "aucune tentative"
    for url, headers in _sources(spec.celex, lang):
        for attempt in range(1, RETRIES + 1):
            try:
                resp = session.get(url, headers=headers, timeout=TIMEOUT_S)
                if resp.status_code == 200 and len(resp.content) >= MIN_BYTES:
                    return resp.content, url
                last_error = f"{url} -> HTTP {resp.status_code}, {len(resp.content)} octets"
            except requests.RequestException as exc:
                last_error = f"{url} -> {exc}"
            log.debug("Tentative %d/%d échouée : %s", attempt, RETRIES, last_error)
            time.sleep(BACKOFF_S)
    raise RuntimeError(last_error)


def count_articles(html: str) -> int:
    """Nombre d'articles distincts détectés (par id, sinon par texte)."""
    ids = set(re.findall(r'id="art_(\d+)"', html))
    if ids:
        return len(ids)
    return len(set(re.findall(r"Article\s+(\d+)", html)))


def load_manifest(path: Path) -> dict[ManifestKey, dict]:
    if not path.exists():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {(e["key"], e["lang"]): e for e in entries}


def save_manifest(path: Path, manifest: dict[ManifestKey, dict]) -> None:
    """Écriture atomique (fichier temporaire puis renommage)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    entries = [manifest[k] for k in sorted(manifest)]
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run(
    corpus: CorpusConfig,
    raw_dir: Path,
    manifest_path: Path,
    *,
    only: Sequence[str] | None = None,
    langs: Sequence[str] | None = None,
    force: bool = False,
    downloader: Downloader = download,
    delay: float = DELAY_S,
) -> int:
    """Télécharge le corpus. Retourne 0 si tout est OK, 1 s'il y a des échecs."""
    manifest = load_manifest(manifest_path)
    failures = 0

    for spec in corpus.texts:
        if only and spec.key not in only:
            continue
        for lang in langs or corpus.languages:
            rel = f"{lang}/{spec.key}_{lang}.html"
            target = raw_dir / rel
            if target.exists() and (spec.key, lang) in manifest and not force:
                log.info("[SKIP] %s/%s déjà téléchargé", spec.key, lang)
                continue

            try:
                content, url = downloader(spec, lang)
            except RuntimeError as exc:
                log.error("[ÉCHEC] %s/%s : %s", spec.key, lang, exc)
                failures += 1
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            n = count_articles(content.decode("utf-8", errors="ignore"))
            ok = n >= spec.expected_articles
            log.log(
                logging.INFO if ok else logging.WARNING,
                "[%s] %s/%s : %d Ko, %d/%d articles",
                "OK" if ok else "À VÉRIFIER", spec.key, lang,
                len(content) // 1024, n, spec.expected_articles,
            )
            manifest[(spec.key, lang)] = {
                "key": spec.key,
                "celex": spec.celex,
                "label": spec.label,
                "lang": lang,
                "file": rel,  # relatif à data/raw : portable d'une machine à l'autre
                "source_url": url,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "articles_detected": n,
                "articles_expected": spec.expected_articles,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            save_manifest(manifest_path, manifest)
            time.sleep(delay)

    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Télécharge le corpus EUR-Lex (FR/EN).")
    parser.add_argument("--only", nargs="+", metavar="KEY", help="clés de textes (ex. rgpd dsa)")
    parser.add_argument("--lang", nargs="+", choices=list(config.LANG_ISO3), help="langues")
    parser.add_argument("--force", action="store_true", help="re-télécharger même si présent")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    corpus = config.load_corpus()
    unknown = set(args.only or []) - {t.key for t in corpus.texts}
    if unknown:
        parser.error(f"clés inconnues : {sorted(unknown)}")

    return run(
        corpus, config.RAW_DIR, config.MANIFEST_PATH,
        only=args.only, langs=args.lang, force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())