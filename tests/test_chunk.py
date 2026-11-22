from eurlex_rag.chunking.chunk import chunk_document, find_references
from eurlex_rag.config import ChunkingConfig, load_chunking


def words(text: str) -> int:  # compteur déterministe : 1 mot = 1 token
    return len(text.split())


CFG = ChunkingConfig(
    tokenizer="approx", single_chunk_max_tokens=60, target_tokens=40, max_tokens=70,
    min_tokens=8, parent_max_tokens=150, definition_titles=("définitions",),
    display_names={"t": {"fr": "TXT"}},
    labels={"recital": {"fr": "Considérant"}, "definition": {"fr": "Définition"}},
)


def pt(label, text, subs=None):
    return {"label": label, "text": text, "sub_points": subs or []}


def para(number, text, points=None, closing=""):
    return {"number": number, "text": text, "points": points or [], "closing_text": closing}


def art(number, paragraphs, title="Titre", chapter=None, section=None, label=None):
    return {
        "number": number, "label": label or f"Article {number}", "title": title,
        "chapter": chapter, "section": section, "paragraphs": paragraphs,
    }


def doc(articles, recitals=()):
    return {"key": "t", "celex": "X", "lang": "fr", "recitals": list(recitals), "articles": articles}


def run(articles, recitals=()):
    return chunk_document(doc(articles, recitals), CFG, words)


def sentence(n, tag=""):
    return " ".join(f"mot{tag}{i}" for i in range(n))


def test_short_article_is_one_chunk_with_breadcrumb():
    chunks, parents = run([art("5", [para("1", "Texte court."), para("2", "Autre.")],
                               title="Principes",
                               chapter={"label": "CHAPITRE II", "title": "Principes"})])
    assert len(chunks) == 1 and chunks[0]["chunk_id"] == "t:fr:art5"
    assert chunks[0]["breadcrumb"] == "TXT > CHAPITRE II — Principes > Article 5 — Principes"
    assert chunks[0]["text_for_embedding"].startswith(chunks[0]["breadcrumb"] + "\n")
    assert chunks[0]["paragraphs"] == ["1", "2"]
    assert chunks[0]["parent_id"] == "t:fr:art5" and parents[0]["usable"] is True


def test_section_is_in_breadcrumb():
    chunks, _ = run([art("12", [para("1", "x")],
                         chapter={"label": "CHAPITRE III", "title": "Droits"},
                         section={"label": "Section 1", "title": "Transparence"})])
    assert "CHAPITRE III — Droits > Section 1 — Transparence > Article 12" in chunks[0]["breadcrumb"]


def test_long_article_split_on_paragraph_boundaries_without_loss():
    paras = [para(str(i), sentence(25, tag=f"p{i}_")) for i in range(1, 7)]
    chunks, _ = run([art("9", paras)])
    assert len(chunks) > 1
    ids = [c["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids) and ids[0] == "t:fr:art9:c1"
    # chaque paragraphe apparaît exactement une fois, en entier
    for i in range(1, 7):
        holders = [c for c in chunks if f"motp{i}_0 " in c["text"] + " "]
        assert len(holders) == 1 and f"motp{i}_24" in holders[0]["text"]
    assert all(c["breadcrumb"].count("§") >= 1 for c in chunks)


def test_oversized_paragraph_is_split_between_points_with_intro_repeated():
    points = [pt(f"{chr(97 + i)})", sentence(18, tag=f"q{i}_")) for i in range(6)]
    chunks, _ = run([art("5", [para("1", "Les données doivent être:", points)])])
    assert len(chunks) > 1
    assert all(c["text"].startswith("1. Les données doivent être:") for c in chunks)
    for i in range(6):
        assert sum(f"motq{i}_0 " in c["text"] + " " for c in chunks) == 1


def test_definitions_article_one_chunk_per_definition():
    points = [pt(f"({i})", sentence(14, tag=f"d{i}_")) for i in range(1, 9)]
    chunks, _ = run([art("4", [para(None, "Aux fins du présent règlement, on entend par:", points)],
                         title="Définitions")])
    assert [c["type"] for c in chunks] == ["definition"] * 8
    assert chunks[0]["chunk_id"] == "t:fr:art4:def1" and chunks[0]["definition_number"] == "1"
    assert chunks[0]["text"].startswith("Aux fins du présent règlement")
    assert chunks[0]["breadcrumb"].endswith("Définition (1)")


def test_recital_chunk():
    chunks, parents = run([], recitals=[{"number": "12", "text": "Un considérant."}])
    assert parents == []
    assert chunks[0]["chunk_id"] == "t:fr:rct12" and chunks[0]["type"] == "recital"
    assert chunks[0]["breadcrumb"] == "TXT > Considérant 12" and chunks[0]["parent_id"] is None


def test_tiny_trailing_chunk_is_merged():
    paras = [para("1", sentence(35, "a")), para("2", sentence(30, "b")), para("3", "Fin courte.")]
    chunks, _ = run([art("8", paras)])
    assert all(c["tokens"] >= CFG.min_tokens for c in chunks)
    assert "Fin courte." in chunks[-1]["text"]


def test_parent_not_usable_when_too_long():
    paras = [para(str(i), sentence(40, f"x{i}_")) for i in range(1, 6)]
    _, parents = run([art("3", paras)])
    assert parents[0]["usable"] is False


def test_find_references():
    text = "conformément à l'article 6, paragraphe 1, à l'article 12a, à l'Article 1er et à l'annexe III"
    assert find_references(text) == ["annex:III", "art:1", "art:12a", "art:6"]
    assert find_references("Article 5(1) and Annex IV") == ["annex:IV", "art:5"]


def test_real_chunking_config_loads():
    cfg = load_chunking()
    assert cfg.min_tokens < cfg.target_tokens <= cfg.max_tokens
    assert cfg.display_names["rgpd"]["fr"] == "RGPD"


def test_run_writes_jsonl_and_checks_alignment(tmp_path):
    import json

    from eurlex_rag.chunking.chunk import run as run_chunking
    from eurlex_rag.config import CorpusConfig, TextSpec

    corpus = CorpusConfig(("fr", "en"), (TextSpec("t", "X", "T", expected_articles=1),))
    for lang in ("fr", "en"):
        d = doc([art("1", [para("1", "Texte.")])]) | {"lang": lang}
        out = tmp_path / "parsed" / lang
        out.mkdir(parents=True)
        (out / f"t_{lang}.json").write_text(json.dumps(d))
    cfg = ChunkingConfig(**{**CFG.__dict__, "labels": {"recital": {"fr": "C", "en": "R"},
                                                       "definition": {"fr": "D", "en": "D"}}})
    assert run_chunking(corpus, cfg, tmp_path / "parsed", tmp_path / "chunks", words) == 0
    lines = (tmp_path / "chunks" / "en" / "t_en.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["chunk_id"] == "t:en:art1"
    assert (tmp_path / "chunks" / "fr" / "t_fr.parents.jsonl").exists()