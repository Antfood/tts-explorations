from .language_config import LanguageConfig
from typing import List, Tuple

class PortugueseConfig(LanguageConfig):
    @property
    def units(self) -> List[str]:
        return [
            "", "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito", "nove",
            "dez", "onze", "doze", "treze", "quatorze", "quinze", "dezesseis", "dezessete", 
            "dezoito", "dezenove"
        ]
    
    @property
    def tens(self) -> List[str]:
        return [
            "", "", "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", 
            "oitenta", "noventa"
        ]
    
    @property
    def hundreds(self) -> List[str]:
        return [
            "", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos", 
            "seiscentos", "setecentos", "oitocentos", "novecentos"
        ]
    
    @property
    def scales(self) -> List[Tuple[str, str]]:
        return [
            ("", ""),
            ("mil", "mil"),
            ("milhão", "milhões"),
            ("bilhão", "bilhões"),
            ("trilhão", "trilhões")
        ]
    
    @property
    def decimal_separator(self) -> str:
        return "vírgula"
    
    @property
    def thousands_separator(self) -> str:
        return "."
    
    @property
    def connector_word(self) -> str:
        return "e"
    
    def format_decimal(self, whole_words: str, decimal_digits: List[str]) -> str:
        return f"{whole_words} {self.decimal_separator} {' '.join(decimal_digits)}"
    
    def format_percentage(self, number_words: str) -> str:
        return f"{number_words} por cento"
    
    def format_ordinal(self, num: int, suffix: str) -> str:
        suffix_lower = suffix.lower()
        if num == 1:
            return "primeiro" if suffix_lower in ['º', 'o'] else "primeira"
        elif num == 2:
            return "segundo" if suffix_lower in ['º', 'o'] else "segunda"
        elif num == 3:
            return "terceiro" if suffix_lower in ['º', 'o'] else "terceira"
        else:
            # For other numbers, we'd need the base number converter
            return f"{num}{suffix}"  # Fallback
    
    def format_time(self, hours: int, minutes: int, hour_words: str, minute_words: str) -> str:
        if hours == 1:
            hour_part = "uma hora"
        else:
            hour_part = f"{hour_words} horas"
        
        if minutes == 0:
            return hour_part
        elif minutes == 15:
            return f"{hour_part} e quinze"
        elif minutes == 30:
            return f"{hour_part} e meia"
        elif minutes == 45:
            return f"{hour_part} e quarenta e cinco"
        else:
            return f"{hour_part} e {minute_words}"
    
    def format_time_am_pm(self, hours: int, minutes: int, hour_words: str, minute_words: str, period: str) -> str:
        # Portuguese doesn't typically use AM/PM, but we can handle it
        return self.format_time(hours, minutes, hour_words, minute_words)
    
    def format_hundreds_special_cases(self, hundreds: int, remainder: int) -> str | None:
        if hundreds == 1:
            if remainder == 0:
                return "cem"  # 100 exactly
            else:
                return "cento"  # 101-199, will be "cento e ..."
        return None  
    
    def format_scale_word(self, quotient_words: str, scale_word: str, is_singular: bool) -> str:
        if scale_word == "mil":
            if is_singular:
                return "mil"
            else:
                return f"{quotient_words} mil"
        else:
            return f"{quotient_words} {scale_word}"
    
    def get_zero_word(self) -> str:
        return "zero"
    
    def get_negative_word(self) -> str:
        return "menos"



