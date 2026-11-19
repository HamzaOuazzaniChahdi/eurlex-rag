"""Tests du parseur.

- Les tests basés sur la fixture utilisent un extrait RÉEL (art. 5 du RGPD, FR).
- Les autres reposent sur la structure décrite dans le rapport sur rgpd_fr.html
  (considérants, chapitres, notes) : à confirmer par `uv run eurlex-parse` sur les vrais fichiers.
"""
from pathlib import Path

from eurlex_rag.config import CorpusConfig, TextSpec
from eurlex_rag.parsing.parse import Article, Paragraph, Recital, parse_html, validate

FIXTURE = Path(__file__).parent / "fixtures" / "rgpd_art5_fr.html"


def _articles(html: str | None = None):
    html = html if html is not None else FIXTURE.read_text(encoding="utf-8")
    return {a.number: a for a in parse_html(html)[1]}


def test_article_5_real_extract():
    art = _articles()["5"]
    assert art.label == "Article 5"
    assert art.title == "Principes relatifs au traitement des données à caractère personnel"
    assert [p.number for p in art.paragraphs] == ["1", "2"]

    p1, p2 = art.paragraphs
    assert p1.text == "Les données à caractère personnel doivent être:"
    assert [pt.label for pt in p1.points] == ["a)", "b)", "c)", "d)", "e)", "f)"]
    assert p1.points[0].text.startswith("traitées de manière licite")
    assert p2.points == [] and "(responsabilité)" in p2.text


def test_truncated_article_6_does_not_crash():
    art = _articles()["6"]
    assert art.title == "Licéité du traitement"
    assert art.paragraphs[0].points[0].label == "a)"


def test_article_premier_label_comes_from_text_number_from_id():
    html = """<div id="art_1"><p class="oj-ti-art">Article premier</p>
      <div class="eli-title"><p class="oj-sti-art">Objet et objectifs</p></div>
      <div id="001.001"><p class="oj-normal">1.   Le présent règlement établit des règles.</p></div></div>"""
    art = _articles(html)["1"]
    assert art.label == "Article premier"
    assert art.paragraphs[0].number == "1"


def test_nested_points_and_closing_text():
    html = """
    <div id="art_4"><p class="oj-ti-art">Article 4</p>
      <div class="eli-title"><p class="oj-sti-art">Définitions</p></div>
      <p class="oj-normal">Aux fins du présent règlement, on entend par:</p>
      <table><tbody><tr><td><p>(1)</p></td><td><p>«X»: texte;</p>
        <table><tbody><tr><td><p>a)</p></td><td><p>sous-point</p></td></tr></tbody></table>
      </td></tr></tbody></table>
      <p class="oj-normal">Texte de clôture.</p></div>"""
    para = _articles(html)["4"].paragraphs[0]  # texte libre sous l'article (cas art_4)
    assert para.number is None
    assert para.text == "Aux fins du présent règlement, on entend par:"
    assert para.points[0].label == "(1)"
    assert para.points[0].sub_points[0].label == "a)"
    assert para.closing_text == "Texte de clôture."


def test_loose_text_and_numbered_paragraphs_keep_document_order():
    html = """<div id="art_7"><p class="oj-ti-art">Article 7</p>
      <p class="oj-normal">Introduction libre.</p>
      <div id="007.001"><p class="oj-normal">1.   Premier.</p></div>
      <div id="007.002"><p class="oj-normal">2.   Second.</p></div></div>"""
    paras = _articles(html)["7"].paragraphs
    assert [(p.number, p.text) for p in paras] == [
        (None, "Introduction libre."), ("1", "Premier."), ("2", "Second."),
    ]


def test_article_without_numbered_paragraphs():
    html = """<div id="art_99"><p class="oj-ti-art">Article 99</p>
      <div class="eli-title"><p class="oj-sti-art">Entrée en vigueur</p></div>
      <p class="oj-normal">Le présent règlement entre en vigueur le vingtième jour.</p></div>"""
    assert _articles(html)["99"].paragraphs[0].text.startswith("Le présent règlement")


def test_footnote_calls_are_removed():
    html = """<div id="art_2"><p class="oj-ti-art">Article 2</p>
      <div id="002.001"><p class="oj-normal">1.   Le présent règlement s'applique.<a id="ntc1-L_1" href="#ntr1-L_1"><span class="oj-super oj-note-tag">1</span></a></p></div></div>"""
    assert _articles(html)["2"].paragraphs[0].text == "Le présent règlement s'applique."


def test_recitals_single_and_multi_paragraph():
    html = """
    <div class="eli-subdivision" id="rct_1"><table><col width="4%"/><col width="96%"/><tbody><tr>
      <td><p class="oj-normal">(1)</p></td>
      <td><p class="oj-normal">La protection des personnes physiques est un droit fondamental.</p></td></tr></tbody></table></div>
    <div class="eli-subdivision" id="rct_19"><table><tbody><tr>
      <td><p class="oj-normal">(19)</p></td>
      <td><p class="oj-normal">Premier alinéa.</p><p class="oj-normal">Second alinéa.</p></td></tr></tbody></table></div>"""
    recitals = parse_html(html)[0]
    assert recitals == [
        Recital("1", "La protection des personnes physiques est un droit fondamental."),
        Recital("19", "Premier alinéa. Second alinéa."),
    ]


def test_chapter_and_section_context_nested():
    html = """<div id="cpt_III"><p class="oj-ti-section-1"><span class="oj-italic">CHAPITRE III</span></p>
      <div class="eli-title" id="cpt_III.tit_1"><p class="oj-ti-section-2">Droits de la personne concernée</p></div>
      <div id="cpt_III.sct_1"><p class="oj-ti-section-1"><span class="oj-expanded">Section 1</span></p>
        <div class="eli-title" id="cpt_III.sct_1.tit_1"><p class="oj-ti-section-2">Transparence et modalités</p></div>
        <div id="art_12"><p class="oj-ti-art">Article 12</p><div id="012.001"><p class="oj-normal">1.   Texte.</p></div></div>
      </div></div>"""
    art = _articles(html)["12"]
    assert art.chapter == {"id": "cpt_III", "label": "CHAPITRE III", "title": "Droits de la personne concernée"}
    assert art.section == {"id": "cpt_III.sct_1", "label": "Section 1", "title": "Transparence et modalités"}


def test_section_of_previous_chapter_is_not_inherited():
    html = """<div id="cpt_I"><p class="oj-ti-section-1">CHAPITRE I</p>
        <div id="cpt_I.sct_1"><p class="oj-ti-section-1">Section 1</p>
          <div id="art_1"><p class="oj-ti-art">Article 1</p><p class="oj-normal">A.</p></div></div></div>
      <div id="cpt_II"><p class="oj-ti-section-1">CHAPITRE II</p>
          <div id="art_2"><p class="oj-ti-art">Article 2</p><p class="oj-normal">B.</p></div></div>"""
    arts = _articles(html)
    assert arts["1"].section["id"] == "cpt_I.sct_1"
    assert arts["2"].chapter["id"] == "cpt_II" and arts["2"].section is None


def test_validate_detects_problems():
    corpus = CorpusConfig(("fr", "en"), (TextSpec("t", "X", "T", expected_articles=3),))
    mk = lambda ns: [Article(str(n), f"Article {n}", "", None, None, [Paragraph("1", "texte")]) for n in ns]
    assert validate({("t", "fr"): mk((1, 2, 3)), ("t", "en"): mk((1, 2, 3))}, corpus) == []
    issues = validate({("t", "fr"): mk((1, 2, 3)), ("t", "en"): mk((1, 3))}, corpus)
    assert any("manquants" in i for i in issues) and any("différente" in i for i in issues)


def test_validate_recitals():
    corpus = CorpusConfig(("fr", "en"), (TextSpec("t", "X", "T", expected_articles=0),))
    fr = [Recital("1", "a"), Recital("2", "b")]
    en = [Recital("1", "a"), Recital("3", "")]
    issues = validate({}, corpus, {("t", "fr"): fr, ("t", "en"): en})
    assert any("considérants manquants" in i for i in issues)
    assert any("considérants vides" in i for i in issues)
    assert any("considérants différente" in i for i in issues)
