"""
Grapheme-to-Phoneme (G2P) module: Converts Chinese text to Phonemes with tones and pause IDs.
"""
from __future__ import annotations
import re
from typing import List, Dict, Tuple


class G2PConverter:
    """Chinese G2P converting normalized text into phoneme IDs with tones."""

    INITIALS = [
        "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
        "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w",
    ]

    def __init__(self):
        # Phoneme vocabulary including special tokens, pause tokens, initials, and finals
        self.vocab = self._build_vocab()
        self.phoneme2id = {p: i for i, p in enumerate(self.vocab)}
        self.id2phoneme = {i: p for i, p in enumerate(self.vocab)}
        # Built-in character to pinyin lookup for core words
        self.char_pinyin_map = {
            "你": "ni3", "好": "hao3", "世": "shi4", "界": "jie4",
            "语": "yu3", "音": "yin1", "合": "he2", "成": "cheng2",
            "大": "da4", "模": "mo2", "型": "xing2", "深": "shen1",
            "度": "du4", "学": "xue2", "习": "xi2", "人": "ren2",
            "工": "gong1", "智": "zhi4", "能": "neng2", "全": "quan2",
            "双": "shuang1", "工": "gong1", "快": "kuai4", "速": "su4",
            "百": "bai3", "分": "fen1", "之": "zhi1", "五": "wu3",
            "十": "shi2", "点": "dian3", "零": "ling2", "一": "yi1",
            "二": "er4", "三": "san1", "四": "si4", "年": "nian2",
        }

    def _build_vocab(self) -> List[str]:
        special = ["<pad>", "<unk>", "<sil>", "<sp>"]
        initials = self.INITIALS
        finals = [
            "a", "o", "e", "i", "u", "v", "ai", "ei", "ui", "ao", "ou",
            "iu", "ie", "ve", "er", "an", "en", "in", "un", "vn", "ang",
            "eng", "ing", "ong",
        ]
        tones = ["1", "2", "3", "4", "5"]
        full_finals = [f + t for f in finals for t in tones]
        return special + initials + full_finals

    def _split_pinyin(self, py: str) -> Tuple[str, str]:
        """Splits pinyin like 'zhong1' -> ('zh', 'ong1') or 'an4' -> ('', 'an4')."""
        match = re.match(r"^([a-z]+)(\d)$", py)
        if not match:
            return "", py
        pinyin_letters, tone = match.group(1), match.group(2)

        for init in sorted(self.INITIALS, key=len, reverse=True):
            if pinyin_letters.startswith(init):
                final = pinyin_letters[len(init):] + tone
                return init, final
        return "", pinyin_letters + tone

    def text_to_phonemes(self, normalized_text: str) -> List[str]:
        """Convert normalized text with prosody tokens into phoneme sequence."""
        tokens = normalized_text.split()
        phonemes = ["<sil>"]

        for tok in tokens:
            if tok == "#2":
                phonemes.append("<sp>")
            elif tok == "#3":
                phonemes.append("<sil>")
            else:
                for char in tok:
                    pinyin = self.char_pinyin_map.get(char, "a1")
                    init, final = self._split_pinyin(pinyin)
                    if init:
                        phonemes.append(init)
                    if final:
                        phonemes.append(final)

        if not phonemes or phonemes[-1] != "<sil>":
            phonemes.append("<sil>")
        return phonemes

    def text_to_ids(self, normalized_text: str) -> List[int]:
        """Convert normalized text directly to sequence of phoneme vocabulary indices."""
        phonemes = self.text_to_phonemes(normalized_text)
        return [self.phoneme2id.get(p, self.phoneme2id["<unk>"]) for p in phonemes]
