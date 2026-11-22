"""Étape 3 — Chunking structurel : data/parsed/*.json -> data/chunks/*.jsonl.

Règles (paramètres dans configs/chunking.yaml) :
  - considérant                         -> 1 chunk
  - article <= single_chunk_max_tokens  -> 1 chunk
  - article de définitions (trop long)  -> 1 définition = 1 chunk (introduction répétée)
  - autre article long                  -> paragraphes regroupés (~target_tokens), jamais coupés ;
                                           un paragraphe trop long est coupé entre ses points,
                                           avec sa phrase d'introduction répétée
  - chunks minuscules (< min_tokens)    -> fusionnés avec un voisin
Chaque chunk porte un fil d'Ariane (texte embeddé = fil d'Ariane + texte) et un parent
(l'article complet), écrit dans <texte>_<lang>.parents.jsonl ; il n'est utilisable par le
LLM que si l'article tient dans parent_max_tokens.

Limites connues : les paragraphes sans points qui dépassent à eux seuls max_tokens ne sont
pas coupés (signalés en avertissement) ; les renvois (`references`) sont détectés par
expressions régulières et incluent aussi les renvois vers d'autres actes.

Usage :
    uv run eurlex-chunk            # tokenizer du modèle d'embeddings (téléchargé au 1er lancement)
    uv run eurlex-chunk --approx   # comptage approximatif, sans réseau
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import re
from collections import Counter as TypeCounter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from eurlex_rag import config
from eurlex_rag.config import ChunkingConfig, CorpusConfig

log = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]
Unit = tuple[list[str], str]  # (numéros de paragraphes couverts, texte)

MIN_DEFINITION_POINTS = 5
_REF_ART = re.compile(r"\b[Aa]rticles?\s+(\d+(?:er)?|\d+[a-z])\b")
_REF_ANNEX = re.compile(r"\b(?:[Aa]nnexe|[Aa]nnex)\s+([IVXL]+)\b")


# ---------------------------------------------------------------- comptage de tokens
def approx_counter(text: str) -> int:
    return math.ceil(len(text) / 3.5)


def load_counter(spec: str) -> TokenCounter:
    if spec == "approx":
        return approx_counter
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(spec)
    except Exception as exc:  # réseau, paquet manquant, identifiant inconnu...
        raise RuntimeError(
            f"Tokenizer '{spec}' indisponible ({exc}). "
            "Installer `tokenizers` et `huggingface_hub`, vérifier le réseau, ou utiliser --approx."
        ) from exc
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


# ---------------------------------------------------------------- rendu du texte
def _render_head(para: dict) -> str:
    number, text = para.get("number"), para.get("text", "")
    if number and text:
        return f"{number}. {text}"
    return text or (f"{number}." if number else "")


def _render_point(point: dict, indent: int = 0) -> str:
    line = ("  " * indent + f"{point.get('label', '')} {point.get('text', '')}").rstrip()
    subs = [_render_point(s, indent + 1) for s in point.get("sub_points", [])]
    return "\n".join([line, *subs])


def _render_paragraph(para: dict) -> str:
    parts = [_render_head(para), *(_render_point(p) for p in para.get("points", []))]
    parts.append(para.get("closing_text", ""))
    return "\n".join(p for p in parts if p)


def find_references(text: str) -> list[str]:
    """Renvois internes approximatifs : 'art:6', 'annex:III'."""
    refs = {f"art:{m.removesuffix('er')}" for m in _REF_ART.findall(text)}
    refs |= {f"annex:{m}" for m in _REF_ANNEX.findall(text)}
    return sorted(refs)


# ---------------------------------------------------------------- fil d'Ariane
def _join_label(label: str | None, title: str | None) -> str:
    return " — ".join(x for x in (label, title) if x)


def _article_breadcrumb(name: str, art: dict) -> str:
    parts = [name]
    for heading in (art.get("chapter"), art.get("section")):
        if heading:
            parts.append(_join_label(heading.get("label"), heading.get("title")))
    parts.append(_join_label(art.get("label"), art.get("title")))
    return " > ".join(p for p in parts if p)


def _para_suffix(numbers: list[str]) -> str:
    if not numbers:
        return ""
    if len(numbers) == 1:
        return f" > §{numbers[0]}"
    return f" > §{numbers[0]}-§{numbers[-1]}"


# ---------------------------------------------------------------- découpage
def _join(units: list[Unit]) -> str:
    return "\n".join(text for _, text in units)


def _unique_numbers(units: list[Unit]) -> list[str]:
    seen: dict[str, None] = {}
    for numbers, _ in units:
        seen.update(dict.fromkeys(numbers))
    return list(seen)


def _split_paragraph(para: dict, tok: TokenCounter, target: int) -> list[str]:
    """Coupe un paragraphe trop long entre ses points ; l'introduction est répétée."""
    head, closing = _render_head(para), para.get("closing_text", "")

    def build(points: list[dict], last: bool) -> str:
        parts = [head, *(_render_point(p) for p in points)]
        if last:
            parts.append(closing)
        return "\n".join(p for p in parts if p)

    pieces, current = [], []
    for point in para["points"]:
        if current and tok(build(current + [point], False)) > target:
            pieces.append(build(current, False))
            current = [point]
        else:
            current.append(point)
    pieces.append(build(current, True))
    return pieces


def _pack(units: list[Unit], tok: TokenCounter, cfg: ChunkingConfig) -> list[Unit]:
    """Regroupe des unités consécutives vers target_tokens, puis fusionne les trop petits."""
    groups: list[list[Unit]] = []
    current: list[Unit] = []
    for unit in units:
        if current and tok(_join(current + [unit])) > cfg.target_tokens:
            groups.append(current)
            current = [unit]
        else:
            current.append(unit)
    if current:
        groups.append(current)

    merged: list[list[Unit]] = []
    for group in groups:
        if (
            merged
            and tok(_join(group)) < cfg.min_tokens
            and tok(_join(merged[-1] + group)) <= cfg.max_tokens
        ):
            merged[-1] = merged[-1] + group
        else:
            merged.append(group)
    if (
        len(merged) > 1
        and tok(_join(merged[0])) < cfg.min_tokens
        and tok(_join(merged[0] + merged[1])) <= cfg.max_tokens
    ):
        merged[1] = merged[0] + merged[1]
        del merged[0]
    return [(_unique_numbers(g), _join(g)) for g in merged]


@dataclass
class _Ctx:
    key: str
    celex: str
    lang: str
    name: str
    cfg: ChunkingConfig
    count: TokenCounter

    def chunk(self, chunk_id: str, type_: str, breadcrumb: str, text: str, **extra) -> dict:
        embedded = f"{breadcrumb}\n{text}"
        record = {
            "chunk_id": chunk_id, "doc_key": self.key, "celex": self.celex, "lang": self.lang,
            "type": type_, "breadcrumb": breadcrumb, "text": text,
            "text_for_embedding": embedded, "tokens": self.count(embedded),
            "article_number": None, "article_label": None, "article_title": None,
            "chapter": None, "section": None, "paragraphs": [], "recital_number": None,
            "definition_number": None, "parent_id": None, "align_key": None,
            "references": find_references(text),
        }
        record.update(extra)
        return record


def _recital_chunk(ctx: _Ctx, recital: dict) -> dict:
    label = f"{ctx.cfg.labels['recital'][ctx.lang]} {recital['number']}"
    return ctx.chunk(
        f"{ctx.key}:{ctx.lang}:rct{recital['number']}", "recital",
        f"{ctx.name} > {label}", recital["text"],
        recital_number=recital["number"], align_key=f"{ctx.key}:rct{recital['number']}",
    )


def _article_chunks(ctx: _Ctx, art: dict) -> tuple[list[dict], dict]:
    cfg, num = ctx.cfg, art["number"]
    pid = f"{ctx.key}:{ctx.lang}:art{num}"
    breadcrumb = _article_breadcrumb(ctx.name, art)
    paras = art["paragraphs"]
    rendered = [_render_paragraph(p) for p in paras]
    full = "\n".join(rendered)

    def tok(text: str) -> int:
        return ctx.count(f"{breadcrumb}\n{text}")

    parent_tokens = tok(full)
    parent = {
        "parent_id": pid, "doc_key": ctx.key, "lang": ctx.lang, "article_number": num,
        "breadcrumb": breadcrumb, "text": full, "tokens": parent_tokens,
        "usable": parent_tokens <= cfg.parent_max_tokens,
    }
    meta = {
        "article_number": num, "article_label": art.get("label"),
        "article_title": art.get("title") or None, "chapter": art.get("chapter"),
        "section": art.get("section"), "parent_id": pid, "align_key": f"{ctx.key}:art{num}",
    }

    para_numbers = [p["number"] for p in paras if p.get("number")]
    if tok(full) <= cfg.single_chunk_max_tokens:
        return [ctx.chunk(pid, "article", breadcrumb, full, paragraphs=para_numbers, **meta)], parent

    is_definitions = (art.get("title") or "").lower() in cfg.definition_titles
    def_label = cfg.labels["definition"][ctx.lang]
    out: list[dict] = []
    pending: list[Unit] = []
    seq = itertools.count(1)

    def flush() -> None:
        for numbers, text in _pack(pending, tok, cfg):
            out.append(ctx.chunk(
                f"{pid}:c{next(seq)}", "article", breadcrumb + _para_suffix(numbers), text,
                paragraphs=numbers, **meta,
            ))
        pending.clear()

    for para, text in zip(paras, rendered):
        numbers = [para["number"]] if para.get("number") else []
        if is_definitions and len(para.get("points", [])) >= MIN_DEFINITION_POINTS:
            flush()
            head = _render_head(para)
            for point in para["points"]:
                label = point.get("label", "")
                slug = re.sub(r"[^0-9A-Za-z]", "", label) or str(len(out) + 1)
                body = "\n".join(x for x in (head, _render_point(point)) if x)
                out.append(ctx.chunk(
                    f"{pid}:def{slug}", "definition", f"{breadcrumb} > {def_label} {label}", body,
                    definition_number=slug, **meta,
                ))
            if para.get("closing_text"):
                pending.append(([], para["closing_text"]))
        elif tok(text) > cfg.max_tokens and para.get("points"):
            pending.extend((numbers, piece) for piece in _split_paragraph(para, tok, cfg.target_tokens))
        else:
            pending.append((numbers, text))
    flush()
    return out, parent


def chunk_document(doc: dict, cfg: ChunkingConfig, count: TokenCounter) -> tuple[list[dict], list[dict]]:
    """Transforme un JSON parsé (un texte, une langue) en (chunks, parents)."""
    ctx = _Ctx(
        key=doc["key"], celex=doc["celex"], lang=doc["lang"],
        name=cfg.display_names.get(doc["key"], {}).get(doc["lang"], doc["key"].upper()),
        cfg=cfg, count=count,
    )
    chunks = [_recital_chunk(ctx, r) for r in doc.get("recitals", []) if r.get("text")]
    parents = []
    for art in doc["articles"]:
        art_chunks, parent = _article_chunks(ctx, art)
        chunks.extend(art_chunks)
        parents.append(parent)
    return chunks, parents


# ---------------------------------------------------------------- orchestration
def _percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    corpus: CorpusConfig,
    cfg: ChunkingConfig,
    parsed_dir: Path,
    chunks_dir: Path,
    count: TokenCounter,
    only: Sequence[str] | None = None,
) -> int:
    issues = 0
    covered: dict[str, dict[str, set[str]]] = {}
    for spec in corpus.texts:
        if only and spec.key not in only:
            continue
        for lang in corpus.languages:
            src = parsed_dir / lang / f"{spec.key}_{lang}.json"
            if not src.exists():
                log.warning("[MANQUANT] %s (lancer eurlex-parse)", src)
                issues += 1
                continue
            doc = json.loads(src.read_text(encoding="utf-8"))
            chunks, parents = chunk_document(doc, cfg, count)
            _write_jsonl(chunks_dir / lang / f"{spec.key}_{lang}.jsonl", chunks)
            _write_jsonl(chunks_dir / lang / f"{spec.key}_{lang}.parents.jsonl", parents)

            ids = [c["chunk_id"] for c in chunks]
            if len(ids) != len(set(ids)):
                log.error("[ERREUR] %s/%s : chunk_id dupliqués", spec.key, lang)
                issues += 1
            articles_with_chunks = {c["article_number"] for c in chunks if c["article_number"]}
            missing = {a["number"] for a in doc["articles"]} - articles_with_chunks
            if missing:
                log.error("[ERREUR] %s/%s : articles sans chunk %s", spec.key, lang, sorted(missing))
                issues += 1
            covered.setdefault(spec.key, {})[lang] = articles_with_chunks

            tokens = [c["tokens"] for c in chunks]
            by_type = dict(TypeCounter(c["type"] for c in chunks))
            usable = sum(p["usable"] for p in parents)
            log.info(
                "[OK] %s/%s : %d chunks %s | tokens p50=%d p95=%d max=%d | parents utilisables %d/%d",
                spec.key, lang, len(chunks), by_type, _percentile(tokens, 0.5),
                _percentile(tokens, 0.95), max(tokens, default=0), usable, len(parents),
            )
            oversize = [c["chunk_id"] for c in chunks if c["tokens"] > cfg.max_tokens]
            if oversize:
                log.warning(
                    "[AVERTISSEMENT] %s/%s : %d chunks > %d tokens (ex. %s)",
                    spec.key, lang, len(oversize), cfg.max_tokens, oversize[:3],
                )

    for key, by_lang in covered.items():
        if len({frozenset(v) for v in by_lang.values()}) > 1:
            log.error("[ERREUR] %s : articles couverts différents entre langues", key)
            issues += 1
    return 1 if issues else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Découpe les textes parsés en chunks.")
    parser.add_argument("--approx", action="store_true", help="comptage approximatif (sans réseau)")
    parser.add_argument("--only", nargs="+", metavar="KEY", help="clés de textes (ex. rgpd dsa)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = config.load_chunking()
    try:
        count = load_counter("approx" if args.approx else cfg.tokenizer)
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.approx:
        log.warning("Comptage approximatif : les tailles seront à refaire avec le vrai tokenizer.")
    return run(config.load_corpus(), cfg, config.PARSED_DIR, config.CHUNKS_DIR, count, args.only)


if __name__ == "__main__":
    raise SystemExit(main())