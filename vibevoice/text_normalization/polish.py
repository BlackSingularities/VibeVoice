# Kopia narrator_text.py z projektu StrefaMilczenia (tekst dla lektora).
# Zmiany wprowadzaj w obu miejscach.
"""Tekst dla lektora: zamiana liczb, dat, skrotow i skrotowcow na pelne slowa
oraz zapis fonetyczny slow obcojezycznych.

Uzywane na dwa sposoby:
- jako lokalny tekst zapasowy, gdy AI jest wylaczone lub zawiodlo,
- jako dodatkowe przejscie po wyniku AI, ktore wylapuje cyfry i skroty,
  ktore model zostawil.

Odmiana jest uproszczona (reguly kontekstowe dla najczestszych przypadkow:
lata, wieki, daty, "lata 60.", "60-tych", procenty, ulamki dziesietne).
Pelna odmiane w kontekscie zdania robi AI - ten modul ma dawac zawsze
czytelny, wymawialny tekst bez cyfr i bez obcej pisowni.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Liczebniki glowne
# ---------------------------------------------------------------------------

_UNITS = ["zero", "jeden", "dwa", "trzy", "cztery", "pięć", "sześć", "siedem", "osiem", "dziewięć"]
_TEENS = [
    "dziesięć", "jedenaście", "dwanaście", "trzynaście", "czternaście", "piętnaście",
    "szesnaście", "siedemnaście", "osiemnaście", "dziewiętnaście",
]
_TENS = [
    "", "", "dwadzieścia", "trzydzieści", "czterdzieści", "pięćdziesiąt",
    "sześćdziesiąt", "siedemdziesiąt", "osiemdziesiąt", "dziewięćdziesiąt",
]
_HUNDREDS = [
    "", "sto", "dwieście", "trzysta", "czterysta", "pięćset",
    "sześćset", "siedemset", "osiemset", "dziewięćset",
]
# (pojedyncza, 2-4, 5+)
_SCALES = [
    ("", "", ""),
    ("tysiąc", "tysiące", "tysięcy"),
    ("milion", "miliony", "milionów"),
    ("miliard", "miliardy", "miliardów"),
    ("bilion", "biliony", "bilionów"),
]


def plural_form(n: int, one: str, few: str, many: str) -> str:
    """Forma rzeczownika po liczebniku: 1 -> one, 2-4 (bez 12-14) -> few, reszta -> many."""
    n = abs(n)
    if n == 1:
        return one
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return few
    return many


def _below_thousand(n: int) -> list[str]:
    words: list[str] = []
    h, rest = divmod(n, 100)
    if h:
        words.append(_HUNDREDS[h])
    if 10 <= rest < 20:
        words.append(_TEENS[rest - 10])
    else:
        t, u = divmod(rest, 10)
        if t:
            words.append(_TENS[t])
        if u:
            words.append(_UNITS[u])
    return words


def cardinal(n: int) -> str:
    """Liczebnik glowny w mianowniku, np. 1960 -> 'tysiąc dziewięćset sześćdziesiąt'."""
    if n == 0:
        return "zero"
    if n < 0:
        return "minus " + cardinal(-n)
    groups: list[int] = []
    while n:
        n, g = divmod(n, 1000)
        groups.append(g)
    if len(groups) > len(_SCALES):
        return " ".join(_UNITS[int(d)] for d in str(n))
    words: list[str] = []
    for idx in range(len(groups) - 1, -1, -1):
        g = groups[idx]
        if not g:
            continue
        if idx == 0:
            words.extend(_below_thousand(g))
            continue
        one, few, many = _SCALES[idx]
        if g == 1:
            words.append(one)
        else:
            words.extend(_below_thousand(g))
            words.append(plural_form(g, one, few, many))
    return " ".join(words)


def digits_spelled(text: str) -> str:
    return " ".join(_UNITS[int(d)] for d in text if d.isdigit())


# ---------------------------------------------------------------------------
# Liczebniki porzadkowe (rodzaj meski, mianownik -> odmiana koncowek)
# ---------------------------------------------------------------------------

_ORD_UNITS = ["", "pierwszy", "drugi", "trzeci", "czwarty", "piąty", "szósty", "siódmy", "ósmy", "dziewiąty"]
_ORD_TEENS = [
    "dziesiąty", "jedenasty", "dwunasty", "trzynasty", "czternasty", "piętnasty",
    "szesnasty", "siedemnasty", "osiemnasty", "dziewiętnasty",
]
_ORD_TENS = [
    "", "", "dwudziesty", "trzydziesty", "czterdziesty", "pięćdziesiąty",
    "sześćdziesiąty", "siedemdziesiąty", "osiemdziesiąty", "dziewięćdziesiąty",
]
_ORD_HUNDREDS = [
    "", "setny", "dwusetny", "trzechsetny", "czterechsetny", "pięćsetny",
    "sześćsetny", "siedemsetny", "osiemsetny", "dziewięćsetny",
]

# Koncowki przypadkow: klucz -> (dla "-y", dla "-i" po miekkim, dla "drugi")
CASES = {
    "nom": ("y", "i", "i"),        # pierwszy / trzeci / drugi
    "gen": ("ego", "iego", "iego"),
    "dat": ("emu", "iemu", "iemu"),
    "loc": ("ym", "im", "im"),
    "fem": ("a", "ia", "a"),
    "fem_gen": ("ej", "iej", "iej"),
    "neut": ("e", "ie", "ie"),
    "pl_gen": ("ych", "ich", "ich"),
    "pl_nom": ("e", "ie", "ie"),
    "pl_dat": ("ym", "im", "im"),
    "pl_inst": ("ymi", "imi", "imi"),
}


def _inflect(word: str, case: str) -> str:
    y_end, i_end, drugi_end = CASES[case]
    if word.endswith("drugi"):
        return word[:-1] + drugi_end
    if word.endswith("i"):
        return word[:-1] + i_end
    if word.endswith("y"):
        return word[:-1] + y_end
    return word


def _ordinal_below_hundred(n: int) -> list[str]:
    if n < 10:
        return [_ORD_UNITS[n]] if n else []
    if n < 20:
        return [_ORD_TEENS[n - 10]]
    t, u = divmod(n, 10)
    return [_ORD_TENS[t]] + ([_ORD_UNITS[u]] if u else [])


def ordinal(n: int, case: str = "nom") -> str:
    """Liczebnik porzadkowy. Jak w polszczyznie: setki/tysiace w latach zostaja
    glowne ('tysiąc dziewięćset'), odmieniaja sie dwie ostatnie cyfry."""
    if n <= 0:
        return cardinal(n)
    words: list[str] = []
    rest = n % 100
    high = n - rest
    if rest == 0:
        # 100, 1000, 1900, 2000 ...
        if n % 1000 == 0:
            thousands = n // 1000
            base = "tysięczny" if thousands == 1 else f"{_compact_cardinal(thousands)}tysięczny"
            if thousands > 1 and thousands < 10:
                base = {2: "dwutysięczny", 3: "trzytysięczny", 4: "czterotysięczny"}.get(thousands, base)
            return _inflect(base, case)
        h = (n % 1000) // 100
        prefix = cardinal(n - n % 1000) if n >= 1000 else ""
        last = _inflect(_ORD_HUNDREDS[h], case)
        return f"{prefix} {last}".strip()
    if high:
        words.append(cardinal(high))
    words.extend(_inflect(w, case) for w in _ordinal_below_hundred(rest))
    return " ".join(words)


def _compact_cardinal(n: int) -> str:
    return cardinal(n).replace(" ", "")


# ---------------------------------------------------------------------------
# Liczby rzymskie (wieki, rzady)
# ---------------------------------------------------------------------------

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(text: str) -> int | None:
    if not text or not re.fullmatch(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})", text):
        return None
    total = 0
    prev = 0
    for ch in reversed(text):
        val = _ROMAN[ch]
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total or None


# ---------------------------------------------------------------------------
# Skroty i skrotowce
# ---------------------------------------------------------------------------

MONTHS_GEN = [
    "", "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
    "lipca", "sierpnia", "września", "października", "listopada", "grudnia",
]

# Skroty z kropka (lub bez) -> pelna forma. Kolejnosc: dluzsze najpierw.
ABBREVIATIONS: list[tuple[str, str]] = [
    (r"m\.in\.", "między innymi"),
    (r"p\.n\.e\.", "przed naszą erą"),
    (r"n\.e\.", "naszej ery"),
    (r"itd\.", "i tak dalej"),
    (r"itp\.", "i tym podobne"),
    (r"tzw\.", "tak zwany"),
    (r"tzn\.", "to znaczy"),
    (r"tj\.", "to jest"),
    (r"np\.", "na przykład"),
    (r"ok\.", "około"),
    (r"ds\.", "do spraw"),
    (r"pt\.", "pod tytułem"),
    (r"wg", "według"),
    (r"im\.", "imienia"),
    (r"ul\.", "ulica"),
    (r"al\.", "aleja"),
    (r"pl\.", "plac"),
    (r"woj\.", "województwo"),
    (r"pow\.", "powiat"),
    (r"gm\.", "gmina"),
    (r"prof\.", "profesor"),
    (r"doc\.", "docent"),
    (r"dr hab\.", "doktor habilitowany"),
    (r"dr", "doktor"),
    (r"inż\.", "inżynier"),
    (r"mgr", "magister"),
    (r"ks\.", "ksiądz"),
    (r"abp", "arcybiskup"),
    (r"bp", "biskup"),
    (r"św\.", "święty"),
    (r"gen\.", "generał"),
    (r"płk", "pułkownik"),
    (r"ppłk", "podpułkownik"),
    (r"mjr", "major"),
    (r"kpt\.", "kapitan"),
    (r"por\.", "porucznik"),
    (r"ppor\.", "podporucznik"),
    (r"sierż\.", "sierżant"),
    (r"kpr\.", "kapral"),
    (r"szer\.", "szeregowy"),
    (r"nr", "numer"),
    (r"s\.", "strona"),
    (r"str\.", "strona"),
    (r"godz\.", "godzina"),
    (r"tys\.", "tysięcy"),
    (r"mln", "milionów"),
    (r"mld", "miliardów"),
    (r"zł", "złotych"),
    (r"proc\.", "procent"),
    (r"km", "kilometrów"),
    (r"kg", "kilogramów"),
    (r"cm", "centymetrów"),
    (r"mm", "milimetrów"),
    (r"tzn", "to znaczy"),
    (r"r\.", "roku"),
    (r"w\.", "wieku"),
    (r"ww\.", "wieków"),
    (r"zm\.", "zmarł"),
    (r"ur\.", "urodzony"),
    (r"jw\.", "jak wyżej"),
    (r"pn\.", "północ"),
    (r"pd\.", "południe"),
    (r"wsch\.", "wschód"),
    (r"zach\.", "zachód"),
]

# Skrotowce czytane litera po literze (polska wymowa liter).
ACRONYMS: dict[str, str] = {
    "IPN": "i-pe-en", "PRL": "pe-er-el", "ZSRR": "zet-es-er-er", "USA": "u-es-a",
    "ONZ": "o-en-zet", "KGB": "ka-gie-be", "SB": "es-be", "UB": "u-be",
    "NKWD": "en-ka-wu-de", "PZPR": "pe-zet-pe-er", "AK": "a-ka", "CIA": "ce-i-a",
    "FBI": "ef-be-i", "UE": "u-e", "RP": "er-pe", "III RP": "trzecia er-pe",
    "PKP": "pe-ka-pe", "PKS": "pe-ka-es", "PAN": "pe-a-en", "MSW": "em-es-wu",
    "MO": "em-o", "ZOMO": "zomo", "KC": "ka-ce", "BBC": "bi-bi-si", "SS": "es-es",
    "NSDAP": "en-es-de-a-pe", "RAF": "raf", "PCK": "pe-ce-ka", "WHO": "wu-ha-o",
    "PiS": "pis", "PO": "pe-o", "PSL": "pe-es-el", "SLD": "es-el-de", "TVP": "te-fau-pe",
    "UJ": "u-jot", "UW": "u-wu", "PWN": "pe-wu-en", "GUS": "gus", "NBP": "en-be-pe",
    "ZUS": "zus", "NFZ": "en-ef-zet", "ABW": "a-be-wu", "CBA": "ce-be-a",
    "MSZ": "em-es-zet", "MON": "mon", "WP": "wu-pe", "LWP": "el-wu-pe",
    "GRU": "gie-er-u", "DDR": "de-de-er", "RFN": "er-ef-en", "NRD": "en-er-de",
    "AI": "a-i", "IT": "i-ti", "DNA": "de-en-a", "UFO": "ufo", "NASA": "nasa",
    "NATO": "nato", "UNESCO": "unesko", "PKB": "pe-ka-be", "VAT": "wat",
}


def _abbr_pattern(abbr: str) -> re.Pattern:
    # Skrot musi stac na granicy slowa; konczaca kropka jest w samym wzorcu.
    # Wielkosc pierwszej litery dowolna ("Nr", "Np."), reszta jak we wzorcu.
    first, rest = abbr[0], abbr[1:]
    head = f"[{first}{first.upper()}]" if first.isalpha() else first
    tail = r"(?![\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ])" if not abbr.endswith(r"\.") else ""
    return re.compile(r"(?<![\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ])" + head + rest + tail)


def _sentence_end_after(text: str, end: int) -> bool:
    """Czy po skrocie z kropka konczy sie zdanie (kropka byla tez koncem zdania)."""
    return bool(re.match(r"\s*(?:$|\n|[A-ZĄĆĘŁŃÓŚŹŻ„\"])", text[end:]))


# Skroty, ktore czesto koncza zdanie ("...i tak dalej. Potem"). Pozostale
# (im., ul., gen., św.) stoja przed nazwami wlasnymi - wielka litera po nich
# nie oznacza konca zdania.
_SENTENCE_FINAL_ABBR = {r"itd\.", r"itp\.", r"r\.", r"w\.", r"ww\.", r"tys\.", r"proc\.", r"jw\.", r"n\.e\.", r"p\.n\.e\."}


def _abbr_replacer(full: str, abbr: str):
    final = abbr in _SENTENCE_FINAL_ABBR
    has_dot = abbr.endswith(r"\.")

    def repl(m: re.Match) -> str:
        word = full[0].upper() + full[1:] if m.group(0)[0].isupper() else full
        if has_dot:
            rest = m.string[m.end():]
            if (final and _sentence_end_after(m.string, m.end())) or re.match(r"[ \t]*(?:$|\n)", rest):
                word += "."
        return word
    return repl


_ABBR_COMPILED = [(_abbr_pattern(a), _abbr_replacer(full, a)) for a, full in ABBREVIATIONS]
_ACRONYM_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in ACRONYMS), key=len, reverse=True)) + r")\b")


# ---------------------------------------------------------------------------
# Glowna zamiana
# ---------------------------------------------------------------------------

_SUFFIX_CASE = {
    "tych": "pl_gen", "ych": "pl_gen", "ich": "pl_gen",
    "te": "pl_nom", "e": "pl_nom",
    "tymi": "pl_inst", "ymi": "pl_inst",
    "tym": "loc", "ym": "loc", "im": "loc",
    "ty": "nom", "y": "nom", "gi": "nom", "ci": "nom", "i": "nom",
    "ta": "fem", "a": "fem", "ga": "fem", "cia": "fem",
    "tego": "gen", "ego": "gen", "go": "gen", "giego": "gen", "ciego": "gen",
    "tej": "fem_gen", "ej": "fem_gen", "giej": "fem_gen", "ciej": "fem_gen",
    "temu": "dat", "emu": "dat", "mu": "dat",
}

# (1, 2-4, 5+, ulamek)
_UNIT_FORMS = {
    "km": ("kilometr", "kilometry", "kilometrów", "kilometra"),
    "kg": ("kilogram", "kilogramy", "kilogramów", "kilograma"),
    "cm": ("centymetr", "centymetry", "centymetrów", "centymetra"),
    "mm": ("milimetr", "milimetry", "milimetrów", "milimetra"),
    "m": ("metr", "metry", "metrów", "metra"),
    "zł": ("złoty", "złote", "złotych", "złotego"),
    "tys": ("tysiąc", "tysiące", "tysięcy", "tysiąca"),
    "mln": ("milion", "miliony", "milionów", "miliona"),
    "mld": ("miliard", "miliardy", "miliardów", "miliarda"),
    "godz": ("godzina", "godziny", "godzin", "godziny"),
    "min": ("minuta", "minuty", "minut", "minuty"),
    "sek": ("sekunda", "sekundy", "sekund", "sekundy"),
}

_LOC_PREPS = {"w", "we", "o", "po", "przy", "na"}


def _word_before(text: str, pos: int) -> str:
    m = re.search(r"([\wąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+)\s*$", text[:pos])
    return m.group(1).lower() if m else ""


def _year_phrase(year: int, before: str, noun: str = "roku") -> str:
    case = "loc" if before in _LOC_PREPS else "gen"
    if noun == "rok":
        case = "nom"
    return f"{ordinal(year, case)} {noun}".strip()


def to_narrator_text(text: str) -> str:
    """Zamienia liczby, daty, skroty i skrotowce na pelne slowa."""
    if not text:
        return ""
    out = text

    # Usuniecie skladni Markdown, ktorej lektor nie czyta.
    out = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", out)
    out = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", out)
    out = re.sub(r"^\s{0,3}#{1,6}\s*", "", out, flags=re.M)
    out = re.sub(r"^\s*[-*+]\s+", "", out, flags=re.M)
    out = re.sub(r"^\s*>\s?", "", out, flags=re.M)
    out = re.sub(r"(\*\*|__|\*|_|`)", "", out)

    # Daty dzienne: 12.03.1987 / 12.03.1987 r.
    def date_repl(m: re.Match) -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return m.group(0)
        return f"{ordinal(d, 'gen')} {MONTHS_GEN[mo]} {ordinal(y, 'gen')} roku"

    out = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s*r\.)?", date_repl, out)

    # "12 marca 1987 r." -> dzien jako porzadkowy w dopelniaczu
    month_names = "|".join(MONTHS_GEN[1:])

    def day_month_repl(m: re.Match) -> str:
        return f"{ordinal(int(m.group(1)), 'gen')} {m.group(2)}"

    out = re.sub(r"\b(\d{1,2})\s+(" + month_names + r")\b", day_month_repl, out)

    # Godziny 14:30
    def time_repl(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        if h > 24 or mi > 59:
            return m.group(0)
        # "o 7:15" / "o godz. 7:15" / "po 14:30" -> "o siódmej piętnaście"
        after_prep = re.search(r"(?:^|[^\wąćęłńóśźż])(?:o|po)\s+(?:godz\.\s*|godzinie\s+)?$", m.string[:m.start()], re.I)
        hour = (ordinal(h, "fem_gen") if after_prep else _inflect(ordinal(h), "fem")) if h else "zero"
        return hour if mi == 0 else f"{hour} {cardinal(mi) if mi >= 10 else 'zero ' + cardinal(mi)}"

    out = re.sub(r"\b(\d{1,2}):(\d{2})\b", time_repl, out)

    # Wieki rzymskie: "XX w.", "XIX wieku", "w XX wieku"
    def century_repl(m: re.Match) -> str:
        n = roman_to_int(m.group(1))
        if not n:
            return m.group(0)
        before = _word_before(out_ref[0], m.start())
        case = "loc" if before in _LOC_PREPS else "gen"
        noun = m.group(2)
        if noun.startswith("w."):
            noun_full = "wieku" + ("." if _sentence_end_after(m.string, m.end()) else "")
        elif noun.startswith("ww."):
            noun_full = "wieków"
            case = "gen"
        else:
            noun_full = noun
        return f"{ordinal(n, case)} {noun_full}"

    out_ref = [out]
    out = re.sub(r"\b([IVXLC]{1,7})\s*(ww\.|w\.|wieku|wiek|stulecia)", century_repl, out)

    # Lata dekady: "lata 60.", "latach 60-tych", "lat 60."
    def decade_repl(m: re.Match) -> str:
        word = m.group(1).lower()
        n = int(m.group(2))
        case = {"lata": "pl_nom", "latach": "loc", "lat": "pl_gen", "latami": "pl_inst", "latom": "pl_dat"}.get(word, "pl_nom")
        if case == "loc":
            # "w latach sześćdziesiątych"
            case = "pl_gen"
        return f"{m.group(1)} {ordinal(n, case)}"

    out = re.sub(r"\b(lata|latach|lat|latami|latom)\s+(\d{2,4})(?:\.|-(?:te|tych|tymi|ych|e))?(?![\d])", decade_repl, out, flags=re.I)

    # Lata z "r." lub "roku": "w 1960 r.", "od 1945 roku"
    def year_repl(m: re.Match) -> str:
        year = int(m.group(1))
        before = _word_before(out_y[0], m.start())
        noun = m.group(2)
        if noun.startswith("r"):
            if noun in ("rok",):
                return _year_phrase(year, before, "rok")
            dot = "." if noun == "r." and _sentence_end_after(m.string, m.end()) else ""
            return _year_phrase(year, before, "roku") + dot
        return m.group(0)

    out_y = [out]
    out = re.sub(r"\b(\d{1,4})\s*(r\.|roku|rok)(?![\wąćęłńóśźż])", year_repl, out)

    # Liczba z koncowka porzadkowa: "60-tych", "3-ci", "2-go", "XX-lecie"
    def suffix_repl(m: re.Match) -> str:
        n = int(m.group(1))
        suf = m.group(2).lower()
        if suf == "lecie":
            return _compact_cardinal(n) + "lecie" if n < 100 else cardinal(n) + " lecie"
        case = _SUFFIX_CASE.get(suf)
        if not case:
            return m.group(0)
        return ordinal(n, case)

    out = re.sub(r"\b(\d+)\s*-\s*([a-ząćęłńóśźż]+)\b", suffix_repl, out)

    # Jednostki po liczbie w poprawnej formie: 1 kilometr, 3 kilometry,
    # 5 kilometrów, 2,5 kilometra (ulamki - dopelniacz l. poj.).
    def unit_repl(m: re.Match) -> str:
        raw = m.group(1).replace(" ", "").replace(" ", "")
        forms = _UNIT_FORMS[m.group(2).lower().rstrip(".")]
        if re.search(r"[.,]", raw):
            noun = forms[3]
        else:
            noun = plural_form(int(raw), forms[0], forms[1], forms[2])
        return f"{_number_words(raw)} {noun}"

    out = re.sub(
        r"(?<![\w,])(\d+(?:,\d+)?)\s*(km|kg|cm|mm|zł|tys\.|mln|mld|godz\.|min\.?|sek\.)(?![\wąćęłńóśźż])",
        lambda m: unit_repl(m) if m.group(2).lower().rstrip(".") in _UNIT_FORMS else m.group(0),
        out,
    )

    # Skroty, ktorych forma zalezy od przyimka: "o godz." -> "o godzinie",
    # "przy ul." -> "przy ulicy".
    out = re.sub(r"\b([oO]|[pP]o|[oO]d|[dD]o) godz\.", lambda m: m.group(1) + {"o": " godzinie", "po": " godzinie", "od": " godziny", "do": " godziny"}[m.group(1).lower()], out)
    out = re.sub(r"\b([pP]rzy|[nN]a|[oO]d|[dD]o|[zZ]|[zZ]e) ul\.", lambda m: m.group(1) + " ulicy", out)

    # Procenty i ulamki
    def percent_repl(m: re.Match) -> str:
        dot = "." if m.group(2) == "proc." and _sentence_end_after(m.string, m.end()) else ""
        return f"{_number_words(m.group(1))} procent{dot}"

    out = re.sub(r"(\d+(?:[.,]\d+)?)\s*(%|proc\.)", percent_repl, out)

    # Zakresy "1939-1945" -> "od ... do ..." czytamy jako "tysiąc ... – tysiąc ..."
    out = re.sub(r"\b(\d{3,4})\s*[-–]\s*(\d{3,4})\b", lambda m: f"{cardinal(int(m.group(1)))} – {cardinal(int(m.group(2)))}", out)

    # Liczby z separatorami tysiecy "1 200 000" / "1.200.000"
    out = re.sub(r"\b\d{1,3}(?:[  .]\d{3})+\b(?![.,]\d)", lambda m: re.sub(r"[  .]", "", m.group(0)), out)

    # Skroty (po liczbach, zeby "r." / "w." przy latach juz byly obsluzone)
    for pattern, full in _ABBR_COMPILED:
        out = pattern.sub(full, out)

    # Skrotowce
    out = _ACRONYM_RE.sub(lambda m: ACRONYMS[m.group(1)], out)

    # Pozostale liczby (calkowite i dziesietne)
    out = re.sub(r"(?<![\w])(\d+(?:,\d+)?)(?![\w])", lambda m: _number_words(m.group(1)), out)
    # Cyfry przyklejone do liter (np. "F16", "T34") - czytamy liczbe osobno
    out = re.sub(r"(\d+)", lambda m: " " + _number_words(m.group(1)) + " ", out)

    # Symbole
    out = out.replace("&", " i ").replace("§", " paragraf ").replace("°C", " stopni Celsjusza").replace("°", " stopni")
    out = out.replace("+", " plus ").replace("=", " równa się ")

    out = re.sub(r"[ \t ]+", " ", out)
    out = re.sub(r" +([,.;:!?])", r"\1", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = to_phonetic(out)
    return "\n".join(line.strip() for line in out.split("\n")).strip()


def _number_words(raw: str) -> str:
    raw = raw.replace(" ", "")
    if re.fullmatch(r"\d+[.,]\d+", raw):
        whole, frac = re.split(r"[.,]", raw)
        frac_words = cardinal(int(frac)) if not frac.startswith("0") else digits_spelled(frac)
        return f"{cardinal(int(whole))} przecinek {frac_words}"
    if raw.isdigit():
        if len(raw) > 1 and raw.startswith("0"):
            return digits_spelled(raw)
        return cardinal(int(raw))
    return raw


# ---------------------------------------------------------------------------
# Zapis fonetyczny slow obcojezycznych
# ---------------------------------------------------------------------------
# Lektor czyta po polsku, wiec kazde slowo obcego pochodzenia zapisujemy tak,
# jak brzmi w polskiej wymowie ("Germanwings" -> "Dżermanłings", "Airbus" ->
# "Erbas"). Slownik ma pierwszenstwo; slowa spoza slownika, ktore wygladaja
# obco, przepuszczamy przez ostrozna heurystyke.

PHONETIC: dict[str, str] = {
    # Lotnictwo i transport
    "Airbus": "Erbas", "Boeing": "Boing", "Lufthansa": "Lufthanza",
    "Germanwings": "Dżermanłings", "Eurowings": "Jurołings",
    "Ryanair": "Rajaner", "Wizz Air": "Łiz Er", "EasyJet": "Izidżet",
    "British Airways": "Brytisz Erłejs", "Air France": "Er Frans",
    "American Airlines": "Ameriken Erlajns", "Delta Air Lines": "Delta Er Lajns",
    "United Airlines": "Junajted Erlajns", "Singapore Airlines": "Singapur Erlajns",
    "Malaysia Airlines": "Malejzja Erlajns", "Embraer": "Embraer",
    "Bombardier": "Bombardije", "Concorde": "Konkord", "Cessna": "Sesna",
    "Gulfstream": "Galfstrim", "Rolls-Royce": "Rols Rojs",
    "Heathrow": "Hitrou", "Gatwick": "Gatłik", "Charles de Gaulle": "Szarl de Gol",
    "cockpit": "kokpit", "blackbox": "blekboks",
    # Technologia
    "Google": "Gugl", "Microsoft": "Majkrosoft", "Apple": "Epl",
    "iPhone": "Ajfon", "Facebook": "Fejsbuk", "YouTube": "Jutiub",
    "Twitter": "Tłiter", "Netflix": "Netfliks", "SpaceX": "Spejs Eks",
    "OpenAI": "Open-ej-aj", "ChatGPT": "Czat-dżi-pi-ti", "Nvidia": "Enwidia",
    "Huawei": "Hłałej", "software": "softłer", "hardware": "hardłer",
    "startup": "startap", "smartphone": "smartfon", "online": "onlajn",
    "offline": "oflajn", "e-mail": "imejl", "email": "imejl",
    "newsletter": "niusleter", "briefing": "brifing", "lockdown": "lokdałn",
    "deadline": "dedlajn",
    # Media i instytucje
    "Reuters": "Rojters", "Associated Press": "Asoszjejted Pres",
    "The Guardian": "De Gardian", "The Times": "De Tajms",
    "New York Times": "Niu Jork Tajms", "Washington Post": "Łoszyngton Poust",
    "Wall Street Journal": "Łol Strit Dżornal", "Bloomberg": "Blumberg",
    "Sky News": "Skaj Nius", "Le Monde": "Le Mąd", "Der Spiegel": "Der Szpigel",
    "White House": "Łajt Hałs", "Downing Street": "Dałning Strit",
    "Scotland Yard": "Skotland Jard", "Wall Street": "Łol Strit",
    # Miejsca
    "New York": "Niu Jork", "Los Angeles": "Los Andżeles", "Chicago": "Szikago",
    "Seattle": "Sietl", "Houston": "Hjuston", "Philadelphia": "Filadelfia",
    "San Francisco": "San Francisko", "Las Vegas": "Las Wegas",
    "Washington": "Łoszyngton", "Birmingham": "Bermingem",
    "Manchester": "Manczester", "Liverpool": "Liwerpul", "Edinburgh": "Edynbra",
    "Glasgow": "Glazgou", "Dublin": "Dablin", "Cambridge": "Kejmbridż",
    "Oxford": "Oksford", "Yorkshire": "Jorkszir", "Marseille": "Marsylia",
    "Bordeaux": "Bordo", "Versailles": "Wersal", "Düsseldorf": "Diseldorf",
    "Dusseldorf": "Diseldorf", "Zürich": "Cirych", "Zurich": "Cirych",
    "Köln": "Kelen", "München": "Monachium", "Rio de Janeiro": "Rio de Żanejro",
    "Buenos Aires": "Błenos Ajres", "Melbourne": "Melbern", "Sydney": "Sydnej",
    # Nazwiska i osoby
    "Shakespeare": "Szekspir", "Churchill": "Czerczil", "Eisenhower": "Ajzenhauer",
    "Roosevelt": "Ruzwelt", "Kennedy": "Kenedy", "Nixon": "Niksn",
    "Reagan": "Rejgan", "Bush": "Busz", "Trump": "Tramp", "Biden": "Bajden",
    "Clinton": "Klinton", "Johnson": "Dżonson", "Thatcher": "Taczer",
    "Macron": "Makron", "Sarkozy": "Sarkozi", "Hollande": "Oland",
    "Mitterrand": "Miteran", "Chirac": "Szirak", "Schröder": "Szreder",
    "Schroeder": "Szreder", "Scholz": "Szolc", "Goebbels": "Gebels",
    "Himmler": "Himler", "Mussolini": "Musolini", "Khrushchev": "Chruszczow",
    "Gorbachev": "Gorbaczow", "Michael": "Majkel", "Jennifer": "Dżenifer",
    "George": "Dżordż", "James": "Dżejms", "Joseph": "Dżozef",
    "Charles": "Czarls", "William": "Łiliam", "Elizabeth": "Elizabet",
    "Matthew": "Matju", "Jeffrey": "Dżefri", "Geoffrey": "Dżefri",
    "Smith": "Smit", "Jones": "Dżouns", "Wright": "Rajt", "Hughes": "Hjuz",
    # Marki i firmy
    "Volkswagen": "Folksfagen", "Mercedes-Benz": "Mercedes Benc",
    "Porsche": "Porsze", "Peugeot": "Peżo", "Renault": "Reno",
    "Citroën": "Sitroen", "Chevrolet": "Szewrolet", "Jaguar": "Dżagłar",
    "Toyota": "Tojota", "Mitsubishi": "Micubiszi", "Hyundai": "Hjundaj",
    "Nestlé": "Nestle", "Unilever": "Juniliwer", "Coca-Cola": "Koka Kola",
    "Starbucks": "Starbaks", "Nike": "Najki", "Goldman Sachs": "Goldman Saks",
    "JPMorgan": "Dżej-Pi Morgan", "Deutsche Bank": "Dojcze Bank",
    "BlackRock": "Blekrok",
    # Wyrazy obce w publicystyce
    "know-how": "nou-hau", "fake news": "fejk nius", "thriller": "triler",
    "whisky": "łiski", "jazz": "dżez", "show": "szoł", "design": "dizajn",
    "manager": "menedżer", "leader": "lider", "image": "imidż",
    "coaching": "kouczing", "catering": "katering", "casting": "kasting",
    "lobby": "lobi", "weekend": "łikend", "cheeseburger": "czizburger",
    "sandwich": "sanłicz",
}

_PL = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
_PHONETIC_LOWER = {key.lower(): value for key, value in PHONETIC.items()}
# Slownik lapie tez krotka polska koncowke fleksyjna ("Airbusa" -> "Erbasa").
_PHONETIC_RE = re.compile(
    r"(?<![\w" + _PL + r"])("
    + "|".join(re.escape(k) for k in sorted(PHONETIC, key=len, reverse=True))
    + r")([a-z" + _PL.lower() + r"]{0,3})(?![\w" + _PL + r"])",
    re.IGNORECASE,
)
# Jedno przejscie po tekscie: najpierw slownik, potem zwykle slowo lacinskie.
_PHONETIC_PASS_RE = re.compile(
    _PHONETIC_RE.pattern + r"|[A-Za-z][A-Za-z'-]*",
    re.IGNORECASE,
)

# Litery i polaczenia obce polszczyznie - sygnal, ze slowo trzeba zapisac fonetycznie.
_FOREIGN_MARKERS = re.compile(
    r"(q|x|v|th|ck|ph|sh|oo|ee|ea|igh|augh|ough|tion|air|ay\b|ey\b|oy\b|wh)",
    re.IGNORECASE,
)
_POLISH_LETTERS = re.compile(r"[" + _PL + r"]")

# Heurystyka angielska -> polski zapis (kolejnosc ma znaczenie).
_PHON_RULES: list[tuple[str, str]] = [
    (r"tion\b", "szyn"),
    (r"ough\b", "ou"),
    (r"augh", "af"),
    (r"igh", "aj"),
    (r"wh", "ł"),
    (r"ph", "f"),
    (r"tch", "cz"),
    (r"th", "t"),
    (r"ck", "k"),
    (r"sch", "sz"),
    (r"sh", "sz"),
    (r"^ch", "cz"),
    (r"qu", "kw"),
    (r"q", "k"),
    (r"x", "ks"),
    (r"oo", "u"),
    (r"ee", "i"),
    (r"ea", "i"),
    (r"ou", "au"),
    (r"air", "er"),
    (r"ai", "ej"),
    (r"ay\b", "ej"),
    (r"ey\b", "ej"),
    (r"oy\b", "oj"),
    (r"^ge", "dże"),
    (r"^gi", "dżi"),
    (r"^ja", "dża"),
    (r"^je", "dże"),
    (r"^jo", "dżo"),
    (r"^ju", "dżu"),
    (r"v", "w"),
    (r"w([aeiouy])", r"ł\1"),
    (r"([^cdrszt])h\b", r"\1"),
]
_PHON_COMPILED = [(re.compile(pat), rep) for pat, rep in _PHON_RULES]


def _match_case(source: str, result: str) -> str:
    if source.isupper() and len(source) > 1:
        return result.upper()
    if source[:1].isupper():
        return result[:1].upper() + result[1:]
    return result


def phonetize_word(word: str) -> str:
    """Ostrozna transkrypcja pojedynczego slowa obcego na polski zapis."""
    low = word.lower()
    if low in _PHONETIC_LOWER:
        return _match_case(word, _PHONETIC_LOWER[low])
    if _POLISH_LETTERS.search(low) or not _FOREIGN_MARKERS.search(low):
        return word
    out = low
    for pattern, replacement in _PHON_COMPILED:
        out = pattern.sub(replacement, out)
    return _match_case(word, out)


def to_phonetic(text: str) -> str:
    """Zapisuje slowa obcojezyczne tak, jak czyta je polski lektor."""
    if not text:
        return ""

    def repl(m: re.Match) -> str:
        # Jedno przejscie: trafienie ze slownika nie wraca juz do heurystyki.
        if m.group(1) is not None:
            return _match_case(m.group(1), _PHONETIC_LOWER[m.group(1).lower()]) + m.group(2)
        return phonetize_word(m.group(0))

    return _PHONETIC_PASS_RE.sub(repl, text)


def phonetic_issues(text: str) -> list[str]:
    """Slowa, ktore nadal wygladaja obco (do podgladu w panelu)."""
    found: list[str] = []
    for m in re.finditer(r"[A-Za-z][A-Za-z'-]{2,}", text or ""):
        word = m.group(0)
        if _POLISH_LETTERS.search(word.lower()):
            continue
        if _FOREIGN_MARKERS.search(word.lower()) and word not in found:
            found.append(word)
    return found[:20]


def leftover_issues(text: str) -> list[str]:
    """Fragmenty, ktorych lektor nie przeczyta poprawnie (do podgladu w panelu)."""
    issues = []
    for m in re.finditer(r"\d+", text or ""):
        issues.append(m.group(0))
    issues.extend(w for w in phonetic_issues(text) if w not in issues)
    return issues[:20]
