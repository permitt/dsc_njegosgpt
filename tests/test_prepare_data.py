import pytest

from prepare_data import (
    _SPACED_RE,
    clean_text,
    cyr_to_lat,
    finalize,
    fix_digraph_case,
    has_cyrillic,
    strip_wikitext,
)


@pytest.mark.parametrize(
    ("cyr", "lat"),
    [
        ("Горски вијенац", "Gorski vijenac"),
        ("Луча микрокозма", "Luča mikrokozma"),
        ("Лажни цар Шћепан Мали", "Lažni car Šćepan Mali"),
        ("ђаво, жена, џеп", "đavo, žena, džep"),
        ("абвгдђежзијклљмнњопрстћуфхцчџш", "abvgdđežzijklljmnnjoprstćufhcčdžš"),
    ],
)
def test_basic_words(cyr, lat):
    assert cyr_to_lat(cyr) == lat


@pytest.mark.parametrize(
    ("cyr", "lat"),
    [
        ("љубав", "ljubav"),
        ("Љубав", "Ljubav"),
        ("ЉУБАВ", "LJUBAV"),
        ("Његош", "Njegoš"),
        ("ЊЕГОШ", "NJEGOŠ"),
        ("Џевердар", "Dževerdar"),
        ("ЏЕВЕРДАР", "DŽEVERDAR"),
        ("ЗАХВАЉУЈЕ", "ZAHVALJUJE"),  # digraph in the middle of an all-caps word
        ("КЊАЗ", "KNJAZ"),
        ("ВОЈВОДА ЊЕГОШ", "VOJVODA NJEGOŠ"),
        ("П. П. ЊЕГОШ", "P. P. NJEGOŠ"),
        ("Љ", "Lj"),  # lone capital: title case
    ],
)
def test_digraph_casing(cyr, lat):
    assert cyr_to_lat(cyr) == lat


def test_uppercase_single_letters():
    assert cyr_to_lat("ЧШЖЋЂ") == "ČŠŽĆĐ"
    assert cyr_to_lat("Ђорђе") == "Đorđe"


def test_punctuation_digits_and_latin_untouched():
    s = "„Ко је оно?“ – рече; (1847.) ’ ... ! : fac-simile, Ritter\n\n"
    out = cyr_to_lat(s)
    for ch in "„“–;()’.!:?,\n1847-":
        assert out.count(ch) == s.count(ch)
    assert out == "„Ko je ono?“ – reče; (1847.) ’ ... ! : fac-simile, Ritter\n\n"


def test_no_cyrillic_left():
    alphabet = "АБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШ"
    text = alphabet + alphabet.lower() + " Слава въ вышнихъ Богу ѣ і ѳ ѵ щ й"
    out = cyr_to_lat(text)
    assert not has_cyrillic(out)
    assert has_cyrillic(text)


def test_pre_reform_letters():
    assert cyr_to_lat("въ вышнихъ") == "v vyšnih"
    assert cyr_to_lat("мѣра") == "mera"
    assert cyr_to_lat("Ѳеодоръ") == "Feodor"
    assert cyr_to_lat("міръ") == "mir"


def test_fix_digraph_case():
    assert fix_digraph_case("ZAHVALjUJE Ljubav") == "ZAHVALJUJE Ljubav"
    assert fix_digraph_case("KNjAZ Njegoš") == "KNJAZ Njegoš"
    assert fix_digraph_case("Nj") == "Nj"


def test_strip_wikitext_keeps_verse_lines():
    src = (
        "{{Навигација глава\n| a\n| b\n}}\n"
        "[[Категорија:Песме]]\n"
        "{{Поезија|\nПрви стих,{{стих|5}}\nдруги стих.<ref>напомена</ref>\n}}\n"
        "== Извор ==\n[http://example.com Извор]\n"
    )
    assert strip_wikitext(src) == "Први стих,\nдруги стих."


def test_strip_wikitext_speaker_label():
    src = "<h4>Владика Данило (сам собом)</h4>\n<blockquote>Виђи врага <br />\nсу седам бињишах</blockquote>"
    assert strip_wikitext(src) == "ВЛАДИКА ДАНИЛО (сам собом)\n\nВиђи врага\nсу седам бињишах"


def test_clean_and_finalize():
    raw = "ВЛАДИКА ДАНИЛО\n\n„Ох, ти  –  куго“ …\n\n\n\n. . . . . . .\nО б ј а в а"
    out, dropped = finalize(clean_text(raw))
    assert out == 'VLADIKA DANILO\n"Oh, ti - kugo" ...\n\nObjava'
    assert not dropped


def test_spaced_regex():
    assert _SPACED_RE.fullmatch("K a r a đ o r đ a")
    assert not _SPACED_RE.fullmatch("i u a")
