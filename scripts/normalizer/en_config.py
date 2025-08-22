from typing import List, Tuple
from .language_config import LanguageConfig

class EnglishConfig(LanguageConfig):
    @property
    def units(self) -> List[str]:
        return [
            "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", 
            "seventeen", "eighteen", "nineteen"
        ]
    
    @property
    def tens(self) -> List[str]:
        return [
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
        ]
    
    @property
    def hundreds(self) -> List[str]:
        return [
            "", "one hundred", "two hundred", "three hundred", "four hundred", "five hundred", 
            "six hundred", "seven hundred", "eight hundred", "nine hundred"
        ]
    
    @property
    def scales(self) -> List[Tuple[str, str]]:
        return [
            ("", ""),
            ("thousand", "thousand"),
            ("million", "million"),
            ("billion", "billion"),
            ("trillion", "trillion")
        ]
    
    @property
    def decimal_separator(self) -> str:
        return "point"
    
    @property
    def thousands_separator(self) -> str:
        return ","
    
    @property
    def connector_word(self) -> str:
        return ""  # English doesn't use connectors like Portuguese
    
    def format_decimal(self, whole_words: str, decimal_digits: List[str]) -> str:
        return f"{whole_words} {self.decimal_separator} {' '.join(decimal_digits)}"
    
    def format_percentage(self, number_words: str) -> str:
        return f"{number_words} percent"
    
    def format_ordinal(self, num: int, suffix: str) -> str:
        # English ordinals would need more complex logic
        return f"{num}{suffix}"  # Fallback
    
    def format_time(self, hours: int, minutes: int, hour_words: str, minute_words: str) -> str:
        # For 24-hour format without AM/PM specified
        if minutes == 0:
            return f"{hour_words} o'clock"
        elif minutes == 15:
            return f"quarter past {hour_words}"
        elif minutes == 30:
            return f"half past {hour_words}"
        elif minutes == 45:
            next_hour = hours + 1 if hours < 23 else 0
            next_hour_words = self.units[next_hour] if next_hour < 20 else self._convert_number_en(next_hour)
            return f"quarter to {next_hour_words}"
        else:
            return f"{hour_words} {minute_words}"
    
    def format_time_am_pm(self, hours: int, minutes: int, hour_words: str, minute_words: str, period: str) -> str:
        # Convert to 12-hour format
        display_hours = hours
        if hours == 0:
            display_hours = 12
        elif hours > 12:
            display_hours = hours - 12
        
        # Recalculate hour_words for 12-hour format
        hour_words_12 = self.units[display_hours] if display_hours < 20 else self._convert_number_en(display_hours)
        
        if minutes == 0:
            return f"{hour_words_12} o'clock {period}"
        elif minutes == 15:
            return f"quarter past {hour_words_12} {period}"
        elif minutes == 30:
            return f"half past {hour_words_12} {period}"
        elif minutes == 45:
            next_hour = display_hours + 1 if display_hours < 12 else 1
            next_hour_words = self.units[next_hour] if next_hour < 20 else self._convert_number_en(next_hour)
            return f"quarter to {next_hour_words} {period}"
        else:
            return f"{hour_words_12} {minute_words} {period}"
    
    def _convert_number_en(self, num: int) -> str:
        """Helper method to convert numbers for English time formatting"""
        if num < 20:
            return self.units[num]
        elif num < 100:
            tens, units = divmod(num, 10)
            result = self.tens[tens]
            if units:
                result += f"-{self.units[units]}"
            return result
        return str(num)  # Fallback for larger numbers
    
    def format_hundreds_special_cases(self, hundreds: int, remainder: int) -> str | None:
        return None  # No special cases for English
    
    def format_scale_word(self, quotient_words: str, scale_word: str, is_singular: bool) -> str:
        return f"{quotient_words} {scale_word}"
    
    def get_zero_word(self) -> str:
        return "zero"
    
    def get_negative_word(self) -> str:
        return "negative"
