from typing import Dict
import re

from .language_config import LanguageConfig
from .port_config import PortugueseConfig
from .en_config import EnglishConfig

class TextNormalizer:
    def __init__(self):
        self.languages: Dict[str, LanguageConfig] = {
            'pt': PortugueseConfig(),
            'en': EnglishConfig()
        }
        
        # Regex patterns for different number formats
        self.patterns = {
            'time_am_pm': re.compile(r'\b(\d{1,2}):(\d{2})\s*(AM|PM)\b', re.IGNORECASE),
            'time': re.compile(r'\b(\d{1,2}):(\d{2})\b'),
            'currency_pt': re.compile(r'R\$\s*(\d{1,3}(?:\.\d{3})*),(\d{2})\b'),
            'currency_en': re.compile(r'\$(\d{1,3}(?:,\d{3})*\.\d{2})\b'),
            'percentage': re.compile(r'\b(\d+(?:[.,]\d+)?)%'),
            'decimal_pt': re.compile(r'\b(\d+),(\d+)\b'),
            'decimal_en': re.compile(r'\b(\d+)\.(\d+)\b'),
            'ordinal_pt': re.compile(r'\b(\d+)([ºª°])\b'),
            'thousands_pt': re.compile(r'\b(\d{1,3}(?:\.\d{3})+)(?![,]\d)\b'),  # Avoid matching if followed by comma
            'thousands_en': re.compile(r'\b(\d{1,3}(?:,\d{3})+)(?![.]\d)\b'),  # Avoid matching if followed by period  
            'integer': re.compile(r'\b(\d+)\b')
        }
    
    def register_language(self, code: str, config: LanguageConfig):
        """Register a new language configuration"""
        self.languages[code] = config
    
    def number_to_words(self, num: int, lang_code: str) -> str:
        """Convert number to words using the specified language configuration"""
        config = self.languages[lang_code]
        
        if num == 0:
            return config.get_zero_word()
        
        # Handle special case for hundreds (like Portuguese "cem")
        if num == 100:
            special_case = config.format_hundreds_special_cases(1, 0)
            if special_case:
                return special_case
        
        if num < 0:
            return f"{config.get_negative_word()} {self.number_to_words(-num, lang_code)}"
        
        return self._convert_number(num, config).strip()
    
    def _convert_number(self, num: int, config: LanguageConfig) -> str:
        if num == 0:
            return ""
        
        if num < 20:
            return config.units[num]
        elif num < 100:
            tens, units = divmod(num, 10)
            result = config.tens[tens]
            if units:
                connector = f" {config.connector_word} " if config.connector_word else "-"
                result += f"{connector}{config.units[units]}"
            return result
        elif num < 1000:
            hundreds, remainder = divmod(num, 100)
            
            # Check for special cases first
            special_case = config.format_hundreds_special_cases(hundreds, remainder)
            if special_case:
                if remainder == 0:
                    return special_case  # e.g., "cem" for 100
                else:
                    # e.g., "cento e vinte e um" for 121
                    connector = f" {config.connector_word} " if config.connector_word else " "
                    return f"{special_case}{connector}{self._convert_number(remainder, config)}"
            
            # Regular hundreds (200+)
            result = config.hundreds[hundreds]
            if remainder:
                connector = f" {config.connector_word} " if config.connector_word else " "
                result += f"{connector}{self._convert_number(remainder, config)}"
            return result
        else:
            # Handle larger numbers with scales
            for i, (singular, plural) in enumerate(reversed(config.scales[1:])):
                scale_value = 1000 ** (len(config.scales) - 1 - i)
                if num >= scale_value:
                    quotient, remainder = divmod(num, scale_value)
                    is_singular = quotient == 1
                    scale_word = singular if is_singular else plural
                    
                    quotient_words = self._convert_number(quotient, config)
                    result = config.format_scale_word(quotient_words, scale_word, is_singular)
                    
                    if remainder:
                        connector = f" {config.connector_word} " if config.connector_word else " "
                        result += f"{connector}{self._convert_number(remainder, config)}"
                    return result
        
        return ""
    
    def _normalize_decimal(self, match, lang_code: str) -> str:
        """Convert decimal numbers to words"""
        config = self.languages[lang_code]
        whole_part = match.group(1)
        decimal_part = match.group(2)
        
        whole_num = int(whole_part) if whole_part else 0
        whole_words = self.number_to_words(whole_num, lang_code)
        
        decimal_digits = [config.units[int(digit)] for digit in decimal_part]
        return config.format_decimal(whole_words, decimal_digits)
    
    def _normalize_percentage(self, match, lang_code: str) -> str:
        """Convert percentages to words"""
        config = self.languages[lang_code]
        num_str = match.group(1)
        
        # Handle decimal percentages
        decimal_sep = ',' if lang_code == 'pt' else '.'
        if decimal_sep in num_str:
            num_clean = num_str.replace(',', '.')
            decimal_match = re.match(r'(\d+)\.(\d+)', num_clean)
            if decimal_match:
                decimal_words = self._normalize_decimal(decimal_match, lang_code)
                return config.format_percentage(decimal_words)
        
        number_words = self.number_to_words(int(float(num_str)), lang_code)
        return config.format_percentage(number_words)
    
    def _normalize_time(self, match, lang_code: str) -> str:
        """Convert time format to words"""
        config = self.languages[lang_code]
        hours = int(match.group(1))
        minutes = int(match.group(2))
        
        hour_words = self.number_to_words(hours, lang_code)
        minute_words = self.number_to_words(minutes, lang_code) if minutes > 0 else ""
        
        return config.format_time(hours, minutes, hour_words, minute_words)
    
    def _normalize_time_am_pm(self, match, lang_code: str) -> str:
        """Convert time format with AM/PM to words"""
        config = self.languages[lang_code]
        hours = int(match.group(1))
        minutes = int(match.group(2))
        period = match.group(3).upper()
        
        hour_words = self.number_to_words(hours, lang_code)
        minute_words = self.number_to_words(minutes, lang_code) if minutes > 0 else ""
        
        return config.format_time_am_pm(hours, minutes, hour_words, minute_words, period)
    
    def _normalize_currency_pt(self, match, lang_code: str) -> str:
        """Convert Portuguese currency to words"""
        thousands_part = match.group(1)  # e.g., "1.250"
        cents_part = match.group(2)      # e.g., "50"
        
        # Convert thousands part (remove dots and convert)
        reais_value = int(thousands_part.replace('.', ''))
        reais_words = self.number_to_words(reais_value, lang_code)
        
        # Convert cents part
        cents_value = int(cents_part)
        cents_words = self.number_to_words(cents_value, lang_code)
        
        return f"R$ {reais_words} reais e {cents_words} centavos"
    
    def _normalize_currency_en(self, match, lang_code: str) -> str:
        """Convert English currency to words"""
        amount_str = match.group(1)  # e.g., "1,250.50"
        
        # Split into dollars and cents
        parts = amount_str.split('.')
        dollars_part = parts[0].replace(',', '')  # "1250"
        cents_part = parts[1]  # "50"
        
        dollars_value = int(dollars_part)
        dollars_words = self.number_to_words(dollars_value, lang_code)
        
        cents_value = int(cents_part)
        cents_words = self.number_to_words(cents_value, lang_code)
        
        return f"${dollars_words} dollars and {cents_words} cents"
    
    def _normalize_ordinal(self, match, lang_code: str) -> str:
        """Convert ordinal numbers to words"""
        config = self.languages[lang_code]
        num = int(match.group(1))
        suffix = match.group(2)
        
        return config.format_ordinal(num, suffix)
    
    def normalize_text(self, text: str, language: str = "pt") -> str:
        """
        Normalize text for TTS using language-specific configurations
        
        Args:
            text: Input text to normalize
            language: Language code ('pt', 'en', etc.)
        
        Returns:
            Normalized text with numbers converted to words
        """
        if language not in self.languages:
            raise ValueError(f"Language '{language}' not supported. Available: {list(self.languages.keys())}")
        
        config = self.languages[language]
        
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Order matters - more specific patterns first
        replacements = [
            (self.patterns['time_am_pm'], lambda m: self._normalize_time_am_pm(m, language)),
            (self.patterns['time'], lambda m: self._normalize_time(m, language)),
        ]
        
        if config.thousands_separator == '.':  
            replacements.extend([
                (self.patterns['currency_pt'], lambda m: self._normalize_currency_pt(m, language)),
                (self.patterns['thousands_pt'], lambda m: self.number_to_words(int(m.group(1).replace('.', '')), language)),
                (self.patterns['decimal_pt'], lambda m: self._normalize_decimal(m, language)),
                (self.patterns['percentage'], lambda m: self._normalize_percentage(m, language)),
                (self.patterns['ordinal_pt'], lambda m: self._normalize_ordinal(m, language)),
            ])
        else:  
            replacements.extend([
                (self.patterns['currency_en'], lambda m: self._normalize_currency_en(m, language)),
                (self.patterns['thousands_en'], lambda m: self.number_to_words(int(m.group(1).replace(',', '')), language)),
                (self.patterns['decimal_en'], lambda m: self._normalize_decimal(m, language)),
                (self.patterns['percentage'], lambda m: self._normalize_percentage(m, language)),
            ])
        
        replacements.append(
            (self.patterns['integer'], lambda m: self.number_to_words(int(m.group(1)), language))
        )
        
        for pattern, replacement_func in replacements:
            text = pattern.sub(replacement_func, text)
        
        return re.sub(r'\s+', ' ', text.strip())
