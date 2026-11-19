"""Étape 2 — Parsing du HTML EUR-Lex (data/raw) en JSON structuré (data/parsed).

Structure produite par texte et par langue :
    recitals  : considérants (rct_*)
    articles  : article > paragraphes numérotés > points (a), (b)... > sous-points

Périmètre actuel : considérants + articles (annexes et visas : étape suivante).
Les appels de notes de bas de page (exposants) sont supprimés du texte.

Usage :
    uv run eurlex-parse
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from eurlex_rag import config
from eurlex_rag.config import CorpusConfig
from eurlex_rag.ingestion.fetch import load_manifest

log = logging.getLogger(__name__)

_ART_ID = re.compile(r"^art_([0-9A-Za-z]+)$")  # exclut art_5.tit_1
_RCT_ID = re.compile(r"^rct_(\d+)$")
_PARA_ID = re.compile(r"^\d{3}\.\d{3}$")  # ex. 005.001 (3 chiffres article . 3 chiffres paragraphe)
_CHAPTER_ID = re.compile(r"^cpt_[^.]+$")  # cpt_II
_SECTION_ID = re.compile(r"^cpt_[^.]+\.sct_[^.]+$")  # cpt_III.sct_1
_PARA_NUM = re.compile(r"^(\d+)\.\s+")

Blocks = list[tuple[str, Tag]]


@dataclass
class Point:
    label: str
    text: str
    sub_points: list[Point] = field(default_factory=list)


@dataclass
class Paragraph:
    number: str | None
    text: str
    points: list[Point] = field(default_factory=list)
    closing_text: str = ""  # texte éventuel placé après la liste de points


@dataclass
class Article:
    number: str  # "5", "12a"... (tiré de l'id HTML)
    label: str  # libellé affiché : "Article 5", "Article premier"
    title: str
    chapter: dict | None
    section: dict | None
    paragraphs: list[Paragraph]


@dataclass
class Recital:
    number: str
    text: str


def clean(text: str) -> str:
    """Normalise les espaces (y compris insécables)."""
    return re.sub(r"\s+", " ", text).strip()


def _walk(container: Tag, split_paragraph_divs: bool = False) -> Iterator[tuple[str, Tag]]:
    """Parcourt les blocs <p> et <table> dans l'ordre, en traversant les <div> neutres.

    Avec split_paragraph_divs, les <div id="005.001"> sont renvoyés tels quels sous le
    type "para" (sinon on descend dedans).
    """
    for child in container.children:
        if not isinstance(child, Tag):
            continue
        classes = child.get("class") or []
        if child.name == "p":
            if "oj-ti-art" not in classes:
                yield "p", child
        elif child.name == "table":
            yield "table", child
        elif child.name == "div" and "eli-title" not in classes:
            if split_paragraph_divs and _PARA_ID.match(child.get("id", "")):
                yield "para", child
            else:
                yield from _walk(child, split_paragraph_divs)


def _parse_table(table: Tag) -> list[Point]:
    """Un tableau = une ou plusieurs lignes (marqueur | texte [+ tableaux imbriqués])."""
    points = []
    for row in table.select(":scope > tbody > tr, :scope > tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 2:
            points.append(Point(label="", text=clean(row.get_text())))
            continue
        parts, subs = [], []
        for kind, tag in _walk(cells[1]):
            if kind == "p":
                parts.append(clean(tag.get_text()))
            else:
                subs.extend(_parse_table(tag))
        points.append(Point(clean(cells[0].get_text()), " ".join(p for p in parts if p), subs))
    return points


def _build_paragraph(blocks: Blocks, strip_number: bool = True) -> Paragraph:
    lead, closing, points = [], [], []
    for kind, tag in blocks:
        if kind == "p":
            text = clean(tag.get_text())
            if text:
                (closing if points else lead).append(text)
        else:
            points.extend(_parse_table(tag))
    text = " ".join(lead)
    number = None
    match = _PARA_NUM.match(text) if strip_number else None
    if match:
        number, text = match.group(1), text[match.end():]
    return Paragraph(number, text, points, " ".join(closing))


def _heading(div: Tag) -> dict:
    label = div.find("p", class_="oj-ti-section-1", recursive=False)
    title_div = div.find("div", class_="eli-title", recursive=False)
    title = title_div.find("p") if title_div else None
    return {
        "id": div["id"],
        "label": clean(label.get_text()) if label else None,
        "title": clean(title.get_text()) if title else None,
    }


def _context(art_div: Tag) -> tuple[dict | None, dict | None]:
    """Chapitre et section d'un article (dernier en-tête rencontré avant lui)."""
    chapter_div = art_div.find_previous("div", id=_CHAPTER_ID)
    chapter = _heading(chapter_div) if chapter_div else None
    section = None
    section_div = art_div.find_previous("div", id=_SECTION_ID)
    # La section doit appartenir au chapitre courant (cpt_II. != cpt_I.)
    if chapter and section_div and section_div["id"].startswith(chapter["id"] + "."):
        section = _heading(section_div)
    return chapter, section


def _parse_article(div: Tag) -> Article:
    number = _ART_ID.match(div["id"]).group(1)
    label_tag = div.find("p", class_="oj-ti-art", recursive=False)
    title_div = div.find("div", class_="eli-title", recursive=False)

    # Paragraphes numérotés (<div id="005.001">) et texte "libre" directement sous l'article
    # (ex. art. 4 : phrase d'introduction + définitions), dans l'ordre du document.
    paragraphs: list[Paragraph] = []
    loose: Blocks = []
    for kind, tag in _walk(div, split_paragraph_divs=True):
        if kind == "para":
            if loose:
                paragraphs.append(_build_paragraph(loose))
                loose = []
            paragraphs.append(_build_paragraph(list(_walk(tag))))
        else:
            loose.append((kind, tag))
    if loose:
        paragraphs.append(_build_paragraph(loose))
    paragraphs = [p for p in paragraphs if p.text or p.points]

    chapter, section = _context(div)
    return Article(
        number=number,
        label=clean(label_tag.get_text()) if label_tag else f"Article {number}",
        title=clean(title_div.get_text()) if title_div else "",
        chapter=chapter,
        section=section,
        paragraphs=paragraphs,
    )


def _render_point(point: Point) -> str:
    parts = [point.label, point.text, *(_render_point(s) for s in point.sub_points)]
    return " ".join(p for p in parts if p)


def _parse_recital(div: Tag) -> Recital:
    """Considérant : tableau (numéro | texte). Les listes internes sont aplaties en texte."""
    number = _RCT_ID.match(div["id"]).group(1)
    blocks: Blocks = []
    for kind, tag in _walk(div):
        if kind != "table":
            blocks.append((kind, tag))
            continue
        for row in tag.select(":scope > tbody > tr, :scope > tr"):
            cells = row.find_all("td", recursive=False)
            blocks.extend(_walk(cells[1]) if len(cells) >= 2 else [("p", row)])
    para = _build_paragraph(blocks, strip_number=False)
    parts = [para.text, *(_render_point(p) for p in para.points), para.closing_text]
    return Recital(number, " ".join(p for p in parts if p))


def parse_html(html: str) -> tuple[list[Recital], list[Article]]:
    soup = BeautifulSoup(html, "lxml")

    # Appels de notes de bas de page (exposants) : ils polluent le texte (« règlement1 »)
    for span in soup.select("span.oj-note-tag"):
        (span.find_parent("a") or span).decompose()

    recitals, seen = [], set()
    for div in soup.find_all("div", id=_RCT_ID):
        recital = _parse_recital(div)
        if recital.number not in seen:
            seen.add(recital.number)
            recitals.append(recital)

    articles, seen = [], set()
    for div in soup.find_all("div", id=_ART_ID):
        article = _parse_article(div)
        if article.number not in seen:  # sécurité contre d'éventuels doublons
            seen.add(article.number)
            articles.append(article)
    return recitals, articles


def _missing_numbers(numbers: list[str]) -> list[int]:
    ints = sorted(int(n) for n in numbers if n.isdigit())
    return sorted(set(range(1, (ints[-1] if ints else 0) + 1)) - set(ints))


def validate(
    articles: dict[tuple[str, str], list[Article]],
    corpus: CorpusConfig,
    recitals: dict[tuple[str, str], list[Recital]] | None = None,
) -> list[str]:
    """Contrôles : comptes, numérotation continue, contenu non vide, alignement FR/EN."""
    issues = []
    recitals = recitals or {}
    for spec in corpus.texts:
        art_numbers, rct_numbers = {}, {}
        for lang in corpus.languages:
            tag = f"{spec.key}/{lang}"
            arts = articles.get((spec.key, lang))
            if arts is not None:
                numbers = [a.number for a in arts]
                art_numbers[lang] = numbers
                if len(numbers) != spec.expected_articles:
                    issues.append(f"{tag}: {len(numbers)} articles (attendu {spec.expected_articles})")
                if missing := _missing_numbers(numbers):
                    issues.append(f"{tag}: articles manquants {missing}")
                if empty := [a.number for a in arts if not a.paragraphs]:
                    issues.append(f"{tag}: articles vides {empty}")
            rcts = recitals.get((spec.key, lang))
            if rcts is not None:
                numbers = [r.number for r in rcts]
                rct_numbers[lang] = numbers
                if not numbers:
                    issues.append(f"{tag}: aucun considérant détecté")
                if missing := _missing_numbers(numbers):
                    issues.append(f"{tag}: considérants manquants {missing}")
                if empty := [r.number for r in rcts if not r.text]:
                    issues.append(f"{tag}: considérants vides {empty}")
        if len({tuple(v) for v in art_numbers.values()}) > 1:
            issues.append(f"{spec.key}: liste d'articles différente entre langues")
        if len({tuple(v) for v in rct_numbers.values()}) > 1:
            issues.append(f"{spec.key}: liste de considérants différente entre langues")
    return issues


def run(corpus: CorpusConfig, raw_dir: Path, parsed_dir: Path, manifest_path: Path) -> int:
    manifest = load_manifest(manifest_path)
    all_articles: dict[tuple[str, str], list[Article]] = {}
    all_recitals: dict[tuple[str, str], list[Recital]] = {}
    for spec in corpus.texts:
        for lang in corpus.languages:
            entry = manifest.get((spec.key, lang))
            if entry is None:
                log.warning("[MANQUANT] %s/%s absent du manifest (lancer eurlex-fetch)", spec.key, lang)
                continue
            html = (raw_dir / entry["file"]).read_text(encoding="utf-8")
            recitals, articles = parse_html(html)
            all_recitals[(spec.key, lang)] = recitals
            all_articles[(spec.key, lang)] = articles
            out = parsed_dir / lang / f"{spec.key}_{lang}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            doc = {
                "key": spec.key, "celex": spec.celex, "label": spec.label, "lang": lang,
                "source_sha256": entry["sha256"],
                "recitals": [asdict(r) for r in recitals],
                "articles": [asdict(a) for a in articles],
            }
            out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info(
                "[OK] %s/%s : %d considérants, %d articles -> %s",
                spec.key, lang, len(recitals), len(articles), out.name,
            )

    issues = validate(all_articles, corpus, all_recitals)
    for issue in issues:
        log.warning("[CONTRÔLE] %s", issue)
    if not issues:
        log.info("Tous les contrôles sont passés.")
    return 1 if issues else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse le HTML EUR-Lex en JSON structuré.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return run(config.load_corpus(), config.RAW_DIR, config.PARSED_DIR, config.MANIFEST_PATH)


if __name__ == "__main__":
    raise SystemExit(main())
