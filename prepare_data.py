"""Build the NjegošGPT training corpus.

Pipeline
--------
1. Download the works of Petar II Petrović Njegoš:
   - sr.wikisource.org (Cyrillic wikitext) for everything it has, and
   - the electronic "Sabrana dela" XML edition from ppnjegos.com for what
     Wikisource lacks (full Ogledalo srpsko, letters, prose, testament, the
     remaining shorter poems).
   Every download is cached under ``data/raw/cache/``; cached files are never
   re-downloaded.
2. Strip markup and write one file per work to ``data/raw/<slug>.txt``
   (original script and punctuation).
3. Clean (whitespace, quotes, dashes, editorial brackets, footnotes),
   transliterate Cyrillic -> Serbian Latin (gajica) and write everything to
   ``data/njegos.txt``.
4. Write ``data/SOURCES.md`` with per-work character counts.

Run with:  uv run prepare_data.py
"""

from __future__ import annotations

import collections
import html
import io
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = RAW_DIR / "cache"
OUT_FILE = DATA_DIR / "njegos.txt"
SOURCES_FILE = DATA_DIR / "SOURCES.md"

USER_AGENT = "njegosgpt-dataset/0.1"
WIKI_RAW = "https://sr.wikisource.org/w/index.php?title={title}&action=raw"
WIKI_VIEW = "https://sr.wikisource.org/wiki/{title}"
PPN_ZIP_URL = "https://ppnjegos.com/Njegos-Sabrana-dela-XML.zip"
PPN_PDF_URL = "https://ppnjegos.com/{volume}.pdf"
REQUEST_DELAY_S = 1.0

# --------------------------------------------------------------------------- #
# Transliteration
# --------------------------------------------------------------------------- #

_CYR_TO_LAT = {
    # Modern Serbian Cyrillic (Vuk's alphabet)
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e",
    "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj",
    "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č",
    "џ": "dž", "ш": "š",
    # Pre-reform / Church Slavonic / Russian letters that show up in quotations
    "ѣ": "e", "і": "i", "ї": "i", "ѳ": "f", "ѵ": "i", "ѕ": "dz", "ѡ": "o",
    "ъ": "", "ь": "", "ы": "y", "й": "j", "щ": "šč", "э": "e", "ю": "ju",
    "я": "ja", "ё": "jo", "є": "je", "ґ": "g", "ѐ": "e", "ѝ": "i",
}  # fmt: skip

# Multi-letter outputs whose capital form depends on context (Lj vs LJ).
_DIGRAPHS = {k: v for k, v in _CYR_TO_LAT.items() if len(v) > 1}


def _is_upper_letter(ch: str) -> bool:
    return ch.isalpha() and ch.isupper()


def cyr_to_lat(text: str) -> str:
    """Transliterate Serbian Cyrillic to Serbian Latin (gajica).

    Uppercase letters that map to digraphs (Љ, Њ, Џ, ...) become title case
    (``Lj``) inside normal words and full caps (``LJ``) when a neighbouring
    letter is also uppercase, e.g. ``Љубав -> Ljubav`` but ``ЉУБАВ -> LJUBAV``.
    Non-Cyrillic characters (Latin, digits, punctuation, whitespace) pass
    through unchanged.
    """
    out: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        low = ch.lower()
        if low not in _CYR_TO_LAT:
            out.append(ch)
            continue
        lat = _CYR_TO_LAT[low]
        if ch == low:
            out.append(lat)
            continue
        if low not in _DIGRAPHS:
            out.append(lat.upper())
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        prv = text[i - 1] if i > 0 else ""
        if _is_upper_letter(nxt) or (not nxt.isalpha() and _is_upper_letter(prv)):
            out.append(lat.upper())
        else:
            out.append(lat[0].upper() + lat[1:])
    return "".join(out)


_CYRILLIC_RE = re.compile(r"[Ѐ-ԯᲀ-᲏ⷠ-ⷿꙀ-ꚟ]")


def has_cyrillic(text: str) -> bool:
    return _CYRILLIC_RE.search(text) is not None


_DIGRAPH_IN_CAPS_RE = re.compile(r"\b\w*(?:Lj|Nj|Dž)\w*\b")


def fix_digraph_case(text: str) -> str:
    """Fix Latin digraph casing in all-caps words: 'ZAHVALjUJE' -> 'ZAHVALJUJE'.

    The XML edition writes digraphs as Unicode ligatures (U+01C8 'ǈ' etc.),
    which NFKC turns into title-case 'Lj' even inside all-caps words.
    """

    def repl(m: re.Match[str]) -> str:
        w = m.group(0)
        rest = re.sub(r"(?<=L|N)j|(?<=D)ž", "", w)
        letters = [c for c in rest if c.isalpha()]
        if len(letters) >= 2 and all(c.isupper() for c in letters):
            return w.upper()
        return w

    return _DIGRAPH_IN_CAPS_RE.sub(repl, text)


# --------------------------------------------------------------------------- #
# Work definitions
# --------------------------------------------------------------------------- #


@dataclass
class Page:
    """One wiki page, or a selection of chapters from one XML volume."""

    title: str  # wiki title, or XML volume name (e.g. 'Njegos-Sabrana-dela-09-Pisma')
    heading: str | None = None  # heading printed above this page's text (wiki only)
    kind: str = "wiki"  # "wiki" or "ppnjegos"
    # ppnjegos only: keep a chapter iff select(normalized_title) is True
    select: Callable[[str], bool] | None = None


@dataclass
class Work:
    slug: str
    title: str  # printed (transliterated) at the top of the work
    pages: list[Page]
    notes: str = ""


def _wiki_pages(titles: list[str]) -> list[Page]:
    # heading = link label shown on the author page (strip disambiguation)
    return [Page(t, re.sub(r"\s*\(Његош\)$", "", t)) for t in titles]


def _norm(s: str) -> str:
    """Normalize a title for matching: Latin, lowercase, letters only."""
    s = unicodedata.normalize("NFKC", cyr_to_lat(s)).lower()
    return re.sub(r"[^a-zčćđšž]", "", s)


_ORDINALS = ["прва", "друга", "трећа", "четврта", "пета", "шеста", "седма", "осма", "девета", "десета"]  # fmt: skip
_ORDINALS_N = ["прво", "друго", "треће", "четврто", "пето"]

WIKI_PUSTINJAK = [
    "„Србин српском роду своме“",
    "Црногорац к свемогућем Богу",
    "Ода на дан рођења сверусијског Императора Николаја Првога",
    "„Одјекуј ми гласе, свиро“",
    "Ода на дан рођења насљедника рускога престола Цесаревича Великога Књаза Александра Николајевића",
    "Књазу Александру Николајевићу Галицину",
    "Степану Дмитријевићу Нечајеву Оберпрокурору Свјатјејшаго Синода",
    "„Нево р’јеко, огледало људства“",
    "„Дуну вјетар јаки пут Русије“",
    "Заробљен Црногорац од виле",
    "Ода султану турскоме",
]
WIKI_EARLY = [
    "Црмничани",
    "Нова пјесна црногорска о војни Русах и Тураках почетој у 1828. году",
    "Мали Радојица (Његош)",
    "Бјелице",
    "Кавајка",
    "На топу са Спужа",
    "Епископу Никифору",
    "„Ко је оно на високом брду“",
    "Србин Србима на части захваљује",
]
WIKI_OTHER = [
    "Кула Ђуришића",
    "Чардак Алексића",
    "Вјерни син ноћи пјева похвалу мислима",
    "Ода сунцу",
    "Ноћ скупља вијека",
    "Полазак Помпеја",
]
# Chapters of XML vol. 1 (Pjesme) already taken from Wikisource (title prefixes,
# Latin). 'NA TOPU SA SPUŽA' occurs twice in the volume (two variants); both skipped.
_PJESME_ON_WIKI = [
    _norm(t)
    for t in [
        "Crmničani", "Nova pjesna crnogorska", "Mali Radojica", "Bjelice", "Kavajka",
        "Na topu sa Spuža", "Episkopu Nikiforu", "Ko je ono na visokom brdu",
        "Srbin Srbima na časti", "Srbin srpskom rodu svome", "Crnogorac k svemogućemu",
        "Oda na dan rođenja sverusijskog", "Odjekuj mi glase", "Oda na dan rođenja nasljednika",
        "Knjazu Aleksandru Nikolajeviću", "Stepanu Dmitrijeviću", "Nevo r'jeko",
        "Dunu vjetar", "Zarobljen Crnogorac", "Oda sultanu", "Pjesna za Vida i Mirčetu",
        "Udarac na Martiniće", "Vuk prijatelj ovči", "Nahije", "Vjerni sin noći",
        "Oda suncu", "Paris i Helena ili noć skuplja vijeka", "Kula Đurišića",
        "Čardak Aleksića", "Polazak Pompeja",
    ]
]  # fmt: skip


def _pjesme_not_on_wiki(title: str) -> bool:
    return not any(title.startswith(p) for p in _PJESME_ON_WIKI)


def _not_glas_kamenstaka(title: str) -> bool:
    return "kamenštaka" not in title and "junačkimdjelima" not in title


WORKS: list[Work] = [
    Work(
        "gorski_vijenac",
        "Горски вијенац",
        [Page("Горски вијенац")],
        notes="Complete (dedication, list of characters, all scenes). Speaker labels "
        "uppercased (parenthetical stage directions after them kept as written); "
        "stage directions kept as plain lines. Rows of dots in the dedication "
        "(lines omitted in the original edition) removed.",
    ),
    Work(
        "luca_mikrokozma",
        "Луча микрокозма",
        [Page("Луча микрокозма")],
        notes="Complete: dedication to S. Milutinović + 6 cantos. Footnotes removed.",
    ),
    Work(
        "lazni_car_scepan_mali",
        "Лажни цар Шћепан Мали",
        [Page("Лажни цар Шћепан Мали")]
        + [Page(f"Лажни цар Шћепан Мали/{i + 1}", f"ДЈЕЈСТВИЈЕ {o.upper()}") for i, o in enumerate(_ORDINALS_N)],
        notes="Complete: preface (prose), list of characters, 5 acts. Author's "
        "footnotes removed. Two short Russian/Church-Slavonic phrases are "
        "transliterated (ъ dropped, ы -> y).",
    ),
    Work(
        "svobodijada",
        "Свободијада",
        [Page(f"Слободијада/{i + 1}", f"ПЈЕСМА {o.upper()}") for i, o in enumerate(_ORDINALS)],
        notes="Complete, 10 cantos. Wikisource titles it 'Слободијада'.",
    ),
    Work(
        "pustinjak_cetinjski",
        "Пустињак цетињски",
        _wiki_pages(WIKI_PUSTINJAK),
        notes="The 1834 collection: all 11 poems as listed on the Wikisource author page.",
    ),
    Work(
        "pjesme_1828_1833",
        "Пјесме 1828-1833",
        _wiki_pages(WIKI_EARLY),
        notes="Early poems. 'Kavajka' is also in Milutinović's Pjevanija (folk-song "
        "style). In 'Nova pjesna crnogorska...' Banašević attributes the last few "
        "lines to Sima Milutinović (editor's note on Wikisource); kept.",
    ),
    Work(
        "lijek_jarosti_turske",
        "Лијек јарости турске",
        [Page("Лијек јарости турске")] + [Page(f"Лијек јарости турске/{i}") for i in (1, 2, 3)],
        notes="Complete: 4 songs (Pjesna za Vida i Mirčetu, Udarac na Martiniće, Vuk prijatelj ovči, Nahije).",
    ),
    Work(
        "kula_djurisica_cardak_aleksica",
        "Кула Ђуришића и Чардак Алексића",
        _wiki_pages(WIKI_OTHER[:2]),
        notes="Both epic songs complete.",
    ),
    Work(
        "ostale_pjesme_wikisource",
        "Остале пјесме",
        _wiki_pages(WIKI_OTHER[2:]),
        notes="Later lyric poems on Wikisource. 'Polazak Pompeja' includes Njegoš's own "
        "prose note on the Pompeii excavation; editorial footnotes removed.",
    ),
    Work(
        "pjesme_ostale_ppnjegos",
        "Пјесме (остале)",
        [Page("Njegos-Sabrana-dela-01-Pjesme", kind="ppnjegos", select=_pjesme_not_on_wiki)],
        notes="All poems of the 'Pjesme' volume that are NOT on Wikisource (about 55, "
        "incl. Misao, Pozdrav rodu, Večernja molitva, Tri dana u Trijestu, Pohod "
        "Pompeja, Plač... Pavla Petrovića, occasional and epigrammatic verses). "
        "Editorial footnotes removed.",
    ),
    Work(
        "ogledalo_srpsko",
        "Огледало српско",
        [Page("Njegos-Sabrana-dela-03-Ogledalo-srpsko", kind="ppnjegos")],
        notes="Complete anthology (1845): Njegoš's dedication 'Sjeni Aleksandra "
        "Puškina', his preface (Predislovije) and closing notes (Primječanije), and "
        "61 epic songs. MOST SONGS ARE FOLK SONGS collected by Njegoš (a few are "
        "possibly by Njegoš himself), not his own poems.",
    ),
    Work(
        "proza",
        "Проза",
        [Page("Njegos-Sabrana-dela-07-Proza", kind="ppnjegos")],
        notes="Žitije Mrđena Nesretnikovića; San na Božić. Prose.",
    ),
    Work(
        "pisma",
        "Писма",
        [Page("Njegos-Sabrana-dela-09-Pisma", kind="ppnjegos")],
        notes="All ~200 letters of the edition (1830-1851). Prose with many "
        "Church-Slavonic/Russian forms of the period; letter numbers removed.",
    ),
    Work(
        "testament",
        "Тестамент",
        [Page("Njegos-Sabrana-dela-11-Testament", kind="ppnjegos")],
        notes="Njegoš's testament (prose).",
    ),
]

EXCLUDED = [
    (
        "Бој на салашу (as a standalone Wikisource poem)",
        (
            "Wikisource lists it as attributed to Njegoš but probably not his. (The song "
            "also appears in Ogledalo srpsko, where it is included as part of the anthology.)"
        ),
    ),
    (
        "Глас каменштака",
        (
            "Only a modern Serbian back-translation of the Italian text survives "
            "(ppnjegos.com vol. 6, njegos.org), not Njegoš's own wording. Excluded."
        ),
    ),
    (
        "Преводи (ppnjegos.com vol. 8)",
        (
            "Njegoš's translations (e.g. from the Iliad via Russian). His wording but not "
            "his work; excluded. Easy to add if wanted."
        ),
    ),
    (
        "Биљежница (ppnjegos.com vol. 10)",
        (
            "Personal notebook: mostly French/Russian/Italian excerpts, addresses, tables "
            "and numbers. Too noisy for a char-level model; excluded."
        ),
    ),
]

# --------------------------------------------------------------------------- #
# Download (cached)
# --------------------------------------------------------------------------- #


def _cache_path(key: str, suffix: str) -> Path:
    safe = re.sub(r"[^\w\-]+", "_", key, flags=re.UNICODE).strip("_")
    return CACHE_DIR / f"{safe}{suffix}"


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _cached_download(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    print(f"  downloading {url}")
    data = _http_get(url)
    if not data.strip():
        raise RuntimeError(f"Empty response for {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    time.sleep(REQUEST_DELAY_S)
    return data


def fetch_wiki(title: str) -> str:
    url = WIKI_RAW.format(title=urllib.parse.quote(title.replace(" ", "_")))
    return _cached_download(url, _cache_path(title, ".wiki")).decode("utf-8")


def fetch_ppnjegos_zip() -> zipfile.ZipFile:
    data = _cached_download(PPN_ZIP_URL, CACHE_DIR / "Njegos-Sabrana-dela-XML.zip")
    return zipfile.ZipFile(io.BytesIO(data))


def source_url(page: Page) -> str:
    if page.kind == "wiki":
        return WIKI_VIEW.format(title=urllib.parse.quote(page.title.replace(" ", "_")))
    return f"{PPN_PDF_URL.format(volume=page.title)} (XML: {PPN_ZIP_URL})"


# --------------------------------------------------------------------------- #
# Wikitext -> plain text
# --------------------------------------------------------------------------- #

# Templates whose first positional argument is the actual text.
_KEEP_FIRST_ARG = {"поезија", "poem", "поем"}
# Templates whose last argument is the text (formatting wrappers).
_KEEP_LAST_ARG = {"font color", "color"}
# Templates that are dropped silently (navigation, notes, line numbers, ...).
_DROP = {
    "стих", "stih", "напомена", "напомене", "википедија", "заглавље", "квалитет",
    "навигација глава", "јв-аутор", "ref", "reflist", "defaultsort",
}  # fmt: skip
unknown_templates: collections.Counter[str] = collections.Counter()


def _split_top_level(s: str) -> list[str]:
    """Split template content on '|' that are not inside nested {{ }} or [[ ]]."""
    parts, depth, buf, i = [], 0, [], 0
    while i < len(s):
        two = s[i : i + 2]
        if two in ("{{", "[["):
            depth += 1
            buf.append(two)
            i += 2
        elif two in ("}}", "]]"):
            depth -= 1
            buf.append(two)
            i += 2
        elif s[i] == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
        else:
            buf.append(s[i])
            i += 1
    parts.append("".join(buf))
    return parts


def expand_templates(s: str) -> str:
    """Replace every {{template}} (nesting-aware) by its kept text or nothing."""
    out, i = [], 0
    while True:
        start = s.find("{{", i)
        if start == -1:
            out.append(s[i:])
            break
        out.append(s[i:start])
        depth, j = 0, start
        while j < len(s):
            if s.startswith("{{", j):
                depth += 1
                j += 2
            elif s.startswith("}}", j):
                depth -= 1
                j += 2
                if depth == 0:
                    break
            else:
                j += 1
        parts = _split_top_level(s[start + 2 : j - 2])
        name = parts[0].strip().lower()
        if name in _KEEP_FIRST_ARG and len(parts) > 1:
            out.append(expand_templates(parts[1]))
        elif name in _KEEP_LAST_ARG and len(parts) > 1:
            out.append(expand_templates(parts[-1]))
        elif name not in _DROP:
            unknown_templates[name] += 1
        i = j
    return "".join(out)


def _link_repl(m: re.Match[str]) -> str:
    target, _, label = m.group(1).partition("|")
    ns = target.split(":", 1)[0].strip().lower() if ":" in target else ""
    if ns in {"категорија", "category", "слика", "file", "датотека", "image", "аутор"}:
        return ""
    return label or target


def _speaker(label: str) -> str:
    """'Владика Данило (сам собом):' -> 'ВЛАДИКА ДАНИЛО (сам собом)'."""
    label = re.sub(r"<[^>]+>", "", label).strip().rstrip(":").strip()
    name, paren, rest = label.partition("(")
    return name.strip().upper() + (f" ({rest}" if paren else "")


def strip_wikitext(src: str) -> str:
    """Wikitext (with embedded HTML) -> plain text, keeping verse line breaks."""
    s = src.replace("\r\n", "\n")
    s = re.sub(r"<!--.*?-->", "", s, flags=re.DOTALL)
    s = re.sub(r"<ref[^>/]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.DOTALL)
    s = re.sub(r"<references\s*/?>", "", s)
    # navigation boxes: centred blocks that consist of internal links
    s = re.sub(r"(?is)<center>(?:(?!</center>).)*?\[\[(?:(?!</center>).)*</center>", "", s)
    # drop "Извор" / "Референце" sections (source credits, footnote lists)
    s = re.sub(r"(?ms)^(=+)\s*(Извор|Референце|Напомене)\s*\1\s*$.*?(?=^==|\Z)", "", s)
    s = expand_templates(s)
    s = re.sub(r"\[\[([^\[\]]*)\]\]", _link_repl, s)
    s = re.sub(r"\[https?://[^\]]*\]", "", s)
    s = re.sub(r"-\{(.*?)\}-", r"\1", s)  # language-converter escapes
    s = re.sub(r"__[A-Z]+__", "", s)
    # Gorski vijenac: speaker labels in <h4> -> own line, uppercase
    s = re.sub(r"(?is)<h4[^>]*>(.*?)</h4>", lambda m: f"\n\n{_speaker(m.group(1))}\n", s)
    s = re.sub(r"(?is)<h[1-6][^>]*>(.*?)</h[1-6]>", r"\n\n\1\n\n", s)
    s = re.sub(r"(?m)^(=+)\s*(.*?)\s*\1\s*$", r"\n\2\n", s)  # == Heading ==
    s = re.sub(r"(?i)<br\s*/?>[ \t]*\n?", "\n", s)
    s = re.sub(r"(?i)</?(blockquote|p|poem|div|center|table|tr|td|hr)\b[^>]*>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("'''", "").replace("''", "")
    s = html.unescape(s)
    lines = [ln.strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln != ";"]  # left over from ";[[Аутор:...]]"
    return _squeeze_blank_lines("\n".join(lines))


# --------------------------------------------------------------------------- #
# ppnjegos.com XML edition -> plain text
# --------------------------------------------------------------------------- #

_SPACED_RE = re.compile(r"(?:[^\W\d_] ){3,}[^\W\d_]")
_XML_PREDEFINED = {"amp", "lt", "gt", "quot", "apos"}
# elements whose content is dropped (editorial notes, letter numbers)
_XML_SKIP = {"footnote", "ref", "label", "img", "page-setup", "footnotes", "endnotes"}


def _dtd_entities(dtd: str) -> dict[str, str]:
    raw = dict(re.findall(r'<!ENTITY\s+([\w.-]+)\s+"([^"]*)"', dtd))

    def resolve(v: str, depth: int = 0) -> str:
        v = re.sub(r"&#x([0-9A-Fa-f]+);", lambda m: chr(int(m.group(1), 16)), v)
        if depth > 10:
            return v
        return re.sub(
            r"&([\w.-]+);",
            lambda m: resolve(raw[m.group(1)], depth + 1) if m.group(1) in raw else m.group(0),
            v,
        )

    return {k: resolve(v) for k, v in raw.items()}


def parse_ppnjegos_xml(xml_text: str, entities: dict[str, str]) -> ET.Element:
    """Parse one volume, expanding the DTD's custom entities (&ch; -> č, ...)."""
    s = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text)

    def ent(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in _XML_PREDEFINED or name not in entities:
            return m.group(0)
        return html.escape(entities[name], quote=False)

    s = re.sub(r"&([\w.-]+);", ent, s)
    return ET.fromstring(s)


def _inline(el: ET.Element) -> str:
    parts = [el.text or ""]
    for child in el:
        if child.tag == "pre":  # wide spaces around letter-spaced words
            parts.append(" ")
        elif child.tag not in _XML_SKIP:
            inner = _inline(child)
            # letter-spaced emphasis ('K a r a đ o r đ a'): <o n="..."> holds the word
            if child.tag == "o" and _SPACED_RE.fullmatch(inner.strip()) and child.get("n"):
                inner = child.get("n", "")
            parts.append(inner)
        parts.append(child.tail or "")
    return "".join(parts)


def _blocks(el: ET.Element, out: list[str]) -> None:
    tag = el.tag
    if tag in _XML_SKIP:
        return
    if tag == "verse":
        out.append(_inline(el).strip())
    elif tag in ("p", "pre"):
        out.append(" ".join(_inline(el).split()))
    elif tag == "title":
        out.append("")
        for child in el:
            _blocks(child, out)
        out.append("")
    elif tag == "speaks":  # not used by the volumes we take, but handle it
        who = el.find("who")
        if who is not None:
            out.append(" ".join(_inline(who).split()).upper())
        for child in el:
            if child.tag != "who":
                _blocks(child, out)
        out.append("")
    elif tag == "table":
        for row in el.iter("tr"):
            out.append(" ".join(" ".join(_inline(td).split()) for td in row))
    else:
        if tag in ("chapter", "section", "part", "verses"):
            out.append("")
        for child in el:
            _blocks(child, out)
        if tag in ("chapter", "section", "part", "verses"):
            out.append("")


def ppnjegos_chapters(root: ET.Element) -> list[tuple[str, ET.Element]]:
    """(normalized title, chapter element) for every chapter of a volume."""
    result = []
    for ch in root.iter("chapter"):
        t = ch.find("title")
        title = " ".join(_inline(t).split()) if t is not None else ""
        result.append((_norm(title), ch))
    return result


def xml_chapter_text(ch: ET.Element) -> str:
    out: list[str] = []
    _blocks(ch, out)
    text = unicodedata.normalize("NFKC", "\n".join(out))
    return _squeeze_blank_lines(fix_digraph_case(text))


class PPNjegos:
    """Lazy access to the cached ppnjegos.com XML zip."""

    def __init__(self) -> None:
        self._zip: zipfile.ZipFile | None = None
        self._entities: dict[str, str] = {}

    def volume(self, name: str) -> ET.Element:
        if self._zip is None:
            self._zip = fetch_ppnjegos_zip()
            self._entities = _dtd_entities(self._zip.read("book.dtd").decode("utf-8"))
        xml_text = self._zip.read(f"{name}.xml").decode("utf-8")
        return parse_ppnjegos_xml(xml_text, self._entities)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def _squeeze_blank_lines(s: str) -> str:
    s = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", s)
    return s.strip("\n")


def extract_work(work: Work, ppn: PPNjegos) -> str:
    """Download (cached) + strip markup for all pages of a work."""
    chunks = []
    for page in work.pages:
        if page.kind == "wiki":
            text = strip_wikitext(fetch_wiki(page.title))
            if page.heading:
                # add the heading unless the page already starts with its title
                first = next((ln for ln in text.split("\n") if ln.strip()), "")
                a, b = _norm(first), _norm(page.heading)
                if not (a and (a.startswith(b[:20]) or b.startswith(a[:20]))):
                    text = f"{page.heading}\n\n{text}"
            chunks.append(text)
        else:
            root = ppn.volume(page.title)
            for title, ch in ppnjegos_chapters(root):
                if page.select is None or page.select(title):
                    chunks.append(xml_chapter_text(ch))
    return "\n\n\n".join(chunks)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

ALLOWED_PUNCT = set(" \n.,;:!?'\"-()")
_CHAR_MAP = str.maketrans(
    {
        "„": '"', "“": '"', "”": '"', "«": '"', "»": '"', "‟": '"', "″": '"',
        "’": "'", "‘": "'", "‚": "'", "`": "'", "´": "'", "ʼ": "'", "′": "'",
        "–": "-", "—": "-", "―": "-", "‒": "-", "‐": "-", "‑": "-", "−": "-",
        " ": " ", " ": " ", " ": " ", "\t": " ",
        "[": "", "]": "", "*": "", "<": "", ">": "", "/": "-",
    }
)  # fmt: skip


def _is_speaker_label(line: str) -> bool:
    letters = [c for c in line.split("(", 1)[0] if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def clean_text(text: str) -> str:
    """Normalize one work's text (Cyrillic or Latin; before transliteration)."""
    s = unicodedata.normalize("NFC", text)
    # drop combining accents (e.g. inverted breve / double grave marking prosody)
    s = unicodedata.normalize("NFC", "".join(c for c in s if not unicodedata.combining(c)))
    s = s.translate(_CHAR_MAP)
    s = s.replace("…", "...")
    s = re.sub(r"(?<![\w])3(?=[а-шђјљњћџ])", "З", s)  # OCR: digit 3 for З
    lines = []
    for ln in s.split("\n"):
        ln = re.sub(r" {2,}", " ", ln).strip()
        if re.fullmatch(r"[.\s]{5,}", ln):  # ". . . . ." omitted-line markers
            continue
        if _SPACED_RE.fullmatch(ln):  # letter-spaced heading 'O b j a v a'
            ln = ln.replace(" ", "")
        ln = re.sub(r"\s+([,;:!?])", r"\1", ln)  # 'riječ ,' -> 'riječ,'
        lines.append(ln)
    # speaker label directly followed by its speech (no blank line), unless the
    # next line is itself a label/heading
    out: list[str] = []
    for i, ln in enumerate(lines):
        if ln == "" and out and _is_speaker_label(out[-1]):
            nxt = next((x for x in lines[i + 1 :] if x), "")
            if not _is_speaker_label(nxt):
                continue
        out.append(ln)
    return _squeeze_blank_lines("\n".join(out))


def finalize(text: str) -> tuple[str, collections.Counter[str]]:
    """Transliterate and drop characters outside the allowed vocabulary."""
    lat = cyr_to_lat(text)
    assert not has_cyrillic(lat), "Cyrillic characters left after transliteration"
    dropped: collections.Counter[str] = collections.Counter()
    keep = []
    for c in lat:
        if (c.isascii() and c.isalnum()) or c in "čćđšžČĆĐŠŽ" or c in ALLOWED_PUNCT:
            keep.append(c)
            continue
        # foreign accented Latin letters (é, à, ü, ...) -> base letter
        base = "".join(b for b in unicodedata.normalize("NFKD", c) if b.isascii() and b.isalpha())
        if base:
            keep.append(base)
        dropped[c] += 1
    out = "\n".join(re.sub(r" {2,}", " ", ln).strip() for ln in "".join(keep).split("\n"))
    return _squeeze_blank_lines(out), dropped


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ppn = PPNjegos()

    results: list[tuple[Work, str]] = []
    all_dropped: collections.Counter[str] = collections.Counter()
    for work in WORKS:
        print(f"[{work.slug}]")
        raw = extract_work(work, ppn)
        (RAW_DIR / f"{work.slug}.txt").write_text(raw + "\n", encoding="utf-8")
        final, dropped = finalize(f"{work.title.upper()}\n\n{clean_text(raw)}")
        all_dropped.update(dropped)
        results.append((work, final))

    corpus = "\n\n\n".join(text for _, text in results) + "\n"
    assert not has_cyrillic(corpus), "Cyrillic left in corpus"
    OUT_FILE.write_text(corpus, encoding="utf-8")

    if unknown_templates:
        print(f"WARNING: unknown wiki templates dropped: {dict(unknown_templates)}")
    if all_dropped:
        print(f"Mapped/dropped rare characters: {dict(all_dropped.most_common())}")

    vocab = sorted(set(corpus))
    counts = collections.Counter(corpus)
    print("\nChars per work:")
    for work, text in results:
        print(f"  {work.slug:34s} {len(text):>9,}")
    print(f"\nTotal chars: {len(corpus):,}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Vocab: {json.dumps(''.join(vocab), ensure_ascii=False)}")
    rare = {c: n for c, n in counts.items() if n < 10}
    if rare:
        print(f"Rare chars (<10 occurrences): {rare}")

    write_sources(results, len(corpus), vocab)
    print(f"\nWrote {OUT_FILE.relative_to(ROOT)} and {SOURCES_FILE.relative_to(ROOT)}")


def write_sources(results: list[tuple[Work, str]], total: int, vocab: list[str]) -> None:
    vocab_str = "".join(vocab).replace("\n", "\\n")
    lines = [
        "# NjegošGPT corpus sources",
        "",
        (
            "Generated by `prepare_data.py`. All texts are by Petar II Petrović Njegoš "
            "(1813-1851, public domain), **except the folk songs in *Ogledalo srpsko*** "
            "(an anthology Njegoš compiled), labelled below."
        ),
        "",
        (
            "Text is ijekavian, in Serbian Latin script (Wikisource texts transliterated "
            "from Cyrillic; the ppnjegos.com edition is already in Latin). Markup-stripped "
            "originals: `data/raw/<slug>.txt`; cached downloads: `data/raw/cache/`."
        ),
        "",
        f"- **Total characters:** {total:,}",
        f"- **Vocabulary size:** {len(vocab)}",
        f"- **Vocabulary:** `{vocab_str}`",
        "",
        "| Work | Slug | Source | Chars | Notes |",
        "|---|---|---|---:|---|",
    ]
    for work, text in results:
        src = "sr.wikisource.org" if work.pages[0].kind == "wiki" else "ppnjegos.com XML"
        title = cyr_to_lat(work.title)
        lines.append(f"| {title} | `{work.slug}` | {src} | {len(text):,} | {work.notes} |")
    lines += [
        "",
        "## Sources and licensing",
        "",
        "- **sr.wikisource.org**: public-domain texts (Wikisource, CC BY-SA for the site itself).",
        (
            "- **ppnjegos.com**: *Petar II Petrović Njegoš: Sabrana dela*, electronic "
            "edition, ed. Aleksandar Lazić; XML by Predrag Janičić and Aleksandar Gogić "
            "(2024), <https://ppnjegos.com/sabdela.htm>. Njegoš's text is public domain, "
            "but the edition is published under **CC BY-ND 4.0** (attribution, no "
            "derivatives). We use it for training data with attribution; check the "
            "license before redistributing `data/njegos.txt` publicly. Run with only "
            "Wikisource sources by deleting the ppnjegos works from `WORKS`."
        ),
        "",
        "## Source URLs",
        "",
    ]
    for work, _ in results:
        lines.append(f"### {cyr_to_lat(work.title)}")
        lines.append("")
        for page in work.pages:
            lines.append(f"- <{source_url(page)}>")
        lines.append("")
    lines += ["## Not included", ""]
    for name, why in EXCLUDED:
        lines.append(f"- **{cyr_to_lat(name)}**: {why}")
    lines += [
        "",
        "## Cleaning",
        "",
        (
            "- Wiki templates, footnotes/refs, navigation, categories, source sections and "
            "verse-number markers removed; XML editorial footnotes and letter numbers "
            "removed. Verse line breaks kept."
        ),
        "- Gorski vijenac speaker labels uppercased (e.g. `VLADIKA DANILO`); stage directions kept as plain lines.",
        (
            "- Quotes normalized to `\"` and `'`, all dashes to `-`, `…` to `...`, `/` to "
            "`-`; combining accent marks, `*`, `<>` and editorial `[ ]` removed (content "
            "kept). Any other non-ASCII, non-Serbian letter is dropped (see the run log)."
        ),
        "- At most one blank line inside a work; works separated by two blank lines.",
        "",
    ]
    SOURCES_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
