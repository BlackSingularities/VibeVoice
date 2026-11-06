"""Normalizacja tekstu przed syntezą mowy."""
from .polish import to_narrator_text, to_phonetic


def normalize_polish(text: str, phonetic: bool = True) -> str:
    """Liczby, daty, godziny, wieki, skróty i skrótowce słownie; opcjonalnie
    obce słowa zapisane polską fonetyką (Germanwings -> Dżermanłings)."""
    out = to_narrator_text(text)
    return to_phonetic(out) if phonetic else out


__all__ = ["normalize_polish", "to_narrator_text", "to_phonetic"]
