from typing import List

class RhymeController:
    def __init__(self):
        self.lyrics_data = []
        self.tamil_vowels = {'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ'}

    def _get_last_word(self, line: str) -> str:
        words = line.strip().split()
        return words[-1] if words else ""

    def _get_vowel_ending(self, word: str) -> str:
        for char in reversed(word):
            if char in self.tamil_vowels:
                return char
        return ""

    def validate_rhyme_scheme(self, lyrics: List[str], rhyme_scheme: str) -> bool:
        if len(lyrics) < len(rhyme_scheme):
            return False
        rhyme_groups = {}
        for i, line in enumerate(lyrics[:len(rhyme_scheme)]):
            last_word = self._get_last_word(line)
            vowel_ending = self._get_vowel_ending(last_word)
            rhyme_letter = rhyme_scheme[i].upper()
            rhyme_groups.setdefault(rhyme_letter, []).append(vowel_ending)
        for endings in rhyme_groups.values():
            if len(endings) > 1 and not all(e == endings[0] for e in endings if e):
                return False
        return True