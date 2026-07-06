"""Deterministic text normalization in front of the ML verbalizer.

The acoustic front-end (ukrainian-word-stress + ipa-uk) understands only
Cyrillic, so Latin tokens used to be dropped or garbled, and digits depended
entirely on the best-effort ML verbalizer -- which latches off for the whole
process after a single load failure. This pass is pure Python, always on, and
runs BEFORE the verbalizer:

  - numbers -> Ukrainian words ("18 В" -> "вісімнадцять вольт",
    "2.6" -> "дві цілих шість десятих"), with a genitive-plural unit wordbook
    for the measurement units common in product specs;
  - Latin tokens -> a Cyrillic reading: consonant-only or short ALL-CAPS
    tokens are spelled letter by letter ("SDS" -> "ес-де-ес"), everything else
    is transliterated by sound ("Procraft" -> "прокрафт").
"""
import re

from num2words import num2words

# units commonly following a number in product specs, as spoken-form tuples
# (one, few, many[, decimal]): "один кілограм / два кілограми / п'ять
# кілограмів / дві цілих шість десятих кілограма". A 1-tuple never changes.
_UNITS = {
    "об/хв": ("оберт за хвилину", "оберти за хвилину", "обертів за хвилину"),
    "ход/хв": ("хід за хвилину", "ходи за хвилину", "ходів за хвилину"),
    "л/хв": ("літр за хвилину", "літри за хвилину", "літрів за хвилину"),
    "м/с": ("метр за секунду", "метри за секунду", "метрів за секунду"),
    "км/год": ("кілометр за годину", "кілометри за годину",
               "кілометрів за годину"),
    "°C": ("градус Цельсія", "градуси Цельсія", "градусів Цельсія"),
    "кВт·год": ("кіловат-година", "кіловат-години", "кіловат-годин"),
    "мА·год": ("міліампер-година", "міліампер-години", "міліампер-годин"),
    "А·год": ("ампер-година", "ампер-години", "ампер-годин"),
    "мАг": ("міліампер-година", "міліампер-години", "міліампер-годин"),
    "кВт": ("кіловат", "кіловати", "кіловат", "кіловата"),
    "Вт": ("ват", "вати", "ват", "вата"),
    "кВ": ("кіловольт", "кіловольти", "кіловольт", "кіловольта"),
    "В": ("вольт", "вольти", "вольт", "вольта"),
    "А": ("ампер", "ампери", "ампер", "ампера"),
    "мм": ("міліметр", "міліметри", "міліметрів", "міліметра"),
    "см": ("сантиметр", "сантиметри", "сантиметрів", "сантиметра"),
    "км": ("кілометр", "кілометри", "кілометрів", "кілометра"),
    "м": ("метр", "метри", "метрів", "метра"),
    "кг": ("кілограм", "кілограми", "кілограмів", "кілограма"),
    "г": ("грам", "грами", "грамів", "грама"),
    "мл": ("мілілітр", "мілілітри", "мілілітрів", "мілілітра"),
    "л": ("літр", "літри", "літрів", "літра"),
    "Гц": ("герц", "герци", "герц", "герца"),
    "кГц": ("кілогерц", "кілогерци", "кілогерц", "кілогерца"),
    "Дж": ("джоуль", "джоулі", "джоулів", "джоуля"),
    "Н·м": ("ньютон-метр", "ньютон-метри", "ньютон-метрів", "ньютон-метра"),
    "Нм": ("ньютон-метр", "ньютон-метри", "ньютон-метрів", "ньютон-метра"),
    "бар": ("бар",),
    "атм": ("атмосфера", "атмосфери", "атмосфер"),
    "год": ("година", "години", "годин"),
    "хв": ("хвилина", "хвилини", "хвилин"),
    "с": ("секунда", "секунди", "секунд"),
    "міс": ("місяць", "місяці", "місяців"),
    "шт": ("штука", "штуки", "штук"),
    "грн": ("гривня", "гривні", "гривень"),
    "%": ("відсоток", "відсотки", "відсотків", "відсотка"),
    "°": ("градус", "градуси", "градусів", "градуса"),
}
# units whose noun is feminine -> the count ends "одна/дві", not "один/два"
_FEM_UNITS = {"кВт·год", "мА·год", "А·год", "мАг", "атм", "год", "хв", "с",
              "міс", "шт", "грн"}
_NUM_UNIT = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(%s)(?=[\s.,;:!?)\"'»]|$)"
    % "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True)))
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
# a number glued to letters ("GP30") must split before conversion
_GLUE = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")

# feminine agreement for "цілих/десятих" ("дві цілих", not "два цілих")
_FEM = {"один": "одна", "два": "дві"}
_DENOM = {1: "десятих", 2: "сотих", 3: "тисячних"}


def _fem(words: str) -> str:
    parts = words.split()
    parts[-1] = _FEM.get(parts[-1], parts[-1])
    return " ".join(parts)


def _num_words(token: str, fem: bool = False) -> str:
    token = token.replace(",", ".")
    if "." in token:
        whole, frac = token.split(".", 1)
        ww = _fem(num2words(int(whole), lang="uk"))
        if len(frac) in _DENOM:
            fw = _fem(num2words(int(frac), lang="uk"))
            return "%s цілих %s %s" % (ww, fw, _DENOM[len(frac)])
        digits = " ".join(num2words(int(d), lang="uk") for d in frac)
        return "%s кома %s" % (ww, digits)
    words = num2words(int(token), lang="uk")
    return _fem(words) if fem else words


def _unit_form(unit: str, token: str) -> str:
    forms = _UNITS[unit]
    if len(forms) == 1:
        return forms[0]
    if "." in token or "," in token:            # decimal -> genitive singular
        return forms[3] if len(forms) > 3 else forms[2]
    n = int(token)
    if n % 100 in (11, 12, 13, 14) or n % 10 in (0, 5, 6, 7, 8, 9):
        return forms[2]
    return forms[0] if n % 10 == 1 else forms[1]


def _num_with_unit(m: re.Match) -> str:
    token, unit = m.group(1), m.group(2)
    fem = unit in _FEM_UNITS
    return "%s %s" % (_num_words(token, fem=fem), _unit_form(unit, token))


# ── Latin -> Cyrillic reading ────────────────────────────────────────────────
_LETTER_NAMES = {
    "a": "а", "b": "бе", "c": "це", "d": "де", "e": "е", "f": "еф", "g": "ґе",
    "h": "аш", "i": "і", "j": "йот", "k": "ка", "l": "ель", "m": "ем",
    "n": "ен", "o": "о", "p": "пе", "q": "ку", "r": "ер", "s": "ес",
    "t": "те", "u": "у", "v": "ве", "w": "дубль-ве", "x": "ікс",
    "y": "ігрек", "z": "зет",
}
_DIGRAPHS = [
    ("sch", "ш"), ("tch", "ч"), ("ch", "ч"), ("sh", "ш"), ("kh", "х"),
    ("ph", "ф"), ("th", "т"), ("ck", "к"), ("qu", "кв"), ("oo", "у"),
    ("ee", "і"), ("ea", "і"), ("ay", "ей"), ("ey", "ей"), ("oy", "ой"),
]
_SINGLES = {
    "a": "а", "b": "б", "d": "д", "e": "е", "f": "ф", "g": "ґ", "h": "х",
    "i": "і", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о",
    "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в",
    "w": "в", "x": "кс", "y": "і", "z": "з",
}
# a few tech words whose by-sound transliteration reads wrong
_WORDS = {"plus": "плюс", "mini": "міні", "max": "макс", "pro": "про",
          "turbo": "турбо", "set": "сет", "kit": "кіт"}
_VOWELS = set("aeiouy")
_LATIN = re.compile(r"[A-Za-z]+")


def _translit_word(word: str) -> str:
    low = word.lower()
    if low in _WORDS:
        return _WORDS[low]
    # abbreviations are spelled out: ALL-CAPS up to 5 letters, or any short
    # token with no vowels to read it by ("SDS", "PWM", "BL")
    if (word.isupper() and len(word) <= 5) or \
            (len(word) <= 4 and not (_VOWELS & set(low))):
        return "-".join(_LETTER_NAMES[ch] for ch in low)
    out, i = [], 0
    while i < len(low):
        for dg, cy in _DIGRAPHS:
            if low.startswith(dg, i):
                out.append(cy)
                i += len(dg)
                break
        else:
            ch = low[i]
            if ch == "c":   # hard/soft c by the following letter
                nxt = low[i + 1:i + 2]
                out.append("с" if nxt in ("e", "i", "y") else "к")
            else:
                out.append(_SINGLES.get(ch, ""))
            i += 1
    return "".join(out)


def latin_to_cyrillic(text: str) -> str:
    return _LATIN.sub(lambda m: _translit_word(m.group(0)), text)


def numbers_to_words(text: str) -> str:
    text = _GLUE.sub(" ", text)
    text = _NUM_UNIT.sub(_num_with_unit, text)
    return _NUM.sub(lambda m: _num_words(m.group(0)), text)


def apply(text: str) -> str:
    """Numbers first (their Cyrillic units must still be intact), then Latin."""
    return latin_to_cyrillic(numbers_to_words(text))
