import json

from eurlex_rag.config import CorpusConfig, TextSpec, load_corpus
from eurlex_rag.ingestion.fetch import count_articles, load_manifest, run

FAKE_HTML = ("<div id=\"art_1\"></div><div id=\"art_2\"></div>" + "x" * 60_000).encode()


def _corpus() -> CorpusConfig:
    spec = TextSpec(key="demo", celex="32000R0001", label="Demo", expected_articles=2)
    return CorpusConfig(languages=("fr", "en"), texts=(spec,))


def test_count_articles_by_id():
    assert count_articles('<div id="art_1"></div><div id="art_2"></div>') == 2


def test_count_articles_fallback_text():
    assert count_articles("Article 1 ... Article 2 ... Article 2") == 2


def test_load_corpus_real_config():
    corpus = load_corpus()
    assert {t.key for t in corpus.texts} >= {"rgpd", "ai_act", "dsa", "data_act"}


def test_run_downloads_then_skips(tmp_path):
    calls = []

    def fake(spec, lang):
        calls.append((spec.key, lang))
        return FAKE_HTML, "http://fake"

    raw, manifest = tmp_path / "raw", tmp_path / "raw" / "manifest.json"
    assert run(_corpus(), raw, manifest, downloader=fake, delay=0) == 0
    assert len(calls) == 2 and (raw / "fr" / "demo_fr.html").exists()
    assert set(load_manifest(manifest)) == {("demo", "fr"), ("demo", "en")}
    assert json.loads(manifest.read_text())[0]["file"].endswith(".html")

    run(_corpus(), raw, manifest, downloader=fake, delay=0)  # 2e passage : rien à refaire
    assert len(calls) == 2


def test_run_reports_failure(tmp_path):
    def broken(spec, lang):
        raise RuntimeError("boom")

    assert run(_corpus(), tmp_path, tmp_path / "m.json", downloader=broken, delay=0) == 1