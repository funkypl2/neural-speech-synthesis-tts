"""
Text Normalization (TN) for Chinese and English TTS.
Converts cardinal numbers, digits, percentages, dates, and currency to spoken forms,
and maps punctuation into prosodic pause boundary tokens.
"""
from __future__ import annotations
import re
from typing import Dict, List


class TextNormalizer:
    """Standard rule-based text normalizer with prosody boundary insertion."""

    DIGIT_MAP = {
        "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
        "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
    }
    UNITS = ["", "十", "百", "千"]

    def __init__(self):
        pass

    def _convert_integer(self, num_str: str) -> str:
        """Converts integer strings (0 - 99999999) to spoken Chinese."""
        val = int(num_str)
        if val == 0:
            return "零"
        if len(num_str) > 8:
            # For long numbers like phone numbers or year dates, read digits one by one
            return "".join(self.DIGIT_MAP.get(c, c) for c in num_str)

        # Handle simple 10-19 colloquially
        if val == 10:
            return "十"
        if 10 < val < 20:
            return "十" + self.DIGIT_MAP[str(val)[1]]

        result = []
        n_len = len(num_str)
        for i, digit in enumerate(num_str):
            d = int(digit)
            unit = self.UNITS[(n_len - 1 - i) % 4]
            if d != 0:
                result.append(self.DIGIT_MAP[digit] + unit)
            else:
                if not result or result[-1] != "零":
                    result.append("零")
        res_str = "".join(result).rstrip("零")
        return res_str

    def normalize(self, text: str) -> str:
        """
        Normalize input raw text into spoken representation with prosody marks:
        - Dates: e.g. 2026年 -> 二零二六年
        - Percentages: e.g. 95.5% -> 百分之九十五点五
        - Numbers: e.g. 100 -> 一百
        - Punctuation to prosody pause marks:
          comma -> #2 (short pause), period/exclamation -> #3 (long sentence pause)
        """
        # 1. Percentages (including decimals like 95.5% or integers like 50%)
        def replace_percentage(match):
            val_str = match.group(1)
            if "." in val_str:
                integ, frac = val_str.split(".")
                spoken_integ = self._convert_integer(integ)
                spoken_frac = "".join(self.DIGIT_MAP.get(c, c) for c in frac)
                return f"百分之{spoken_integ}点{spoken_frac}"
            else:
                return f"百分之{self._convert_integer(val_str)}"

        text = re.sub(r"(\d+(?:\.\d+)?)%", replace_percentage, text)

        # 2. Years (4 consecutive digits followed by 年)
        text = re.sub(
            r"(\d{4})年",
            lambda m: "".join(self.DIGIT_MAP.get(c, c) for c in m.group(1)) + "年",
            text,
        )

        # 3. Decimals
        def replace_decimal(match):
            integ, frac = match.group(1), match.group(2)
            spoken_integ = self._convert_integer(integ)
            spoken_frac = "".join(self.DIGIT_MAP.get(c, c) for c in frac)
            return f"{spoken_integ}点{spoken_frac}"

        text = re.sub(r"(\d+)\.(\d+)", replace_decimal, text)

        # 4. Standard cardinal numbers
        text = re.sub(r"\d+", lambda m: self._convert_integer(m.group(0)), text)

        # 5. Punctuation prosody normalization
        # #1: syllable/word boundary, #2: clause pause, #3: sentence pause
        text = re.sub(r"[，,、]", " #2 ", text)
        text = re.sub(r"[。！？!?；;\n]+", " #3 ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

