import pytest
from scripts.normalizer.port_config import PortugueseConfig
from scripts.normalizer.en_config import EnglishConfig


class TestPortugueseConfig:
    def setup_method(self):
        self.config = PortugueseConfig()
    
    def test_basic_properties(self):
        assert self.config.units[1] == "um"
        assert self.config.units[10] == "dez"
        assert self.config.tens[2] == "vinte"
        assert self.config.hundreds[1] == "cento"
        assert self.config.connector_word == "e"
        assert self.config.decimal_separator == "vírgula"
        assert self.config.thousands_separator == "."
    
    def test_scales(self):
        scales = self.config.scales
        assert scales[1] == ("mil", "mil")
        assert scales[2] == ("milhão", "milhões")
    
    def test_format_decimal(self):
        result = self.config.format_decimal("três", ["um", "quatro"])
        assert result == "três vírgula um quatro"
    
    def test_format_percentage(self):
        result = self.config.format_percentage("cinquenta")
        assert result == "cinquenta por cento"
    
    def test_format_ordinal_masculine(self):
        assert self.config.format_ordinal(1, "º") == "primeiro"
        assert self.config.format_ordinal(2, "º") == "segundo"
        assert self.config.format_ordinal(3, "º") == "terceiro"
        assert self.config.format_ordinal(4, "º") == "4º"  # Fallback
    
    def test_format_ordinal_feminine(self):
        assert self.config.format_ordinal(1, "ª") == "primeira"
        assert self.config.format_ordinal(2, "ª") == "segunda"
        assert self.config.format_ordinal(3, "ª") == "terceira"
    
    def test_format_time_basic(self):
        result = self.config.format_time(1, 0, "uma", "")
        assert result == "uma hora"
        
        result = self.config.format_time(2, 0, "duas", "")
        assert result == "duas horas"
    
    def test_format_time_special_minutes(self):
        result = self.config.format_time(2, 15, "duas", "quinze")
        assert result == "duas horas e quinze"
        
        result = self.config.format_time(2, 30, "duas", "trinta")
        assert result == "duas horas e meia"
        
        result = self.config.format_time(2, 45, "duas", "quarenta e cinco")
        assert result == "duas horas e quarenta e cinco"
    
    def test_format_time_regular_minutes(self):
        result = self.config.format_time(3, 25, "três", "vinte e cinco")
        assert result == "três horas e vinte e cinco"
    
    def test_format_hundreds_special_cases(self):
        # Test 100 exactly
        result = self.config.format_hundreds_special_cases(1, 0)
        assert result == "cem"
        
        # Test 101-199
        result = self.config.format_hundreds_special_cases(1, 50)
        assert result == "cento"
        
        # Test other hundreds (no special case)
        result = self.config.format_hundreds_special_cases(2, 0)
        assert result is None
    
    def test_format_scale_word_mil(self):
        # Singular thousand
        result = self.config.format_scale_word("", "mil", True)
        assert result == "mil"
        
        # Plural thousand
        result = self.config.format_scale_word("dois", "mil", False)
        assert result == "dois mil"
    
    def test_format_scale_word_other(self):
        result = self.config.format_scale_word("dois", "milhões", False)
        assert result == "dois milhões"
    
    def test_get_words(self):
        assert self.config.get_zero_word() == "zero"
        assert self.config.get_negative_word() == "menos"


class TestEnglishConfig:
    def setup_method(self):
        self.config = EnglishConfig()
    
    def test_basic_properties(self):
        assert self.config.units[1] == "one"
        assert self.config.units[10] == "ten"
        assert self.config.tens[2] == "twenty"
        assert self.config.hundreds[1] == "one hundred"
        assert self.config.connector_word == ""  # No connector in English
        assert self.config.decimal_separator == "point"
        assert self.config.thousands_separator == ","
    
    def test_scales(self):
        scales = self.config.scales
        assert scales[1] == ("thousand", "thousand")
        assert scales[2] == ("million", "million")
    
    def test_format_decimal(self):
        result = self.config.format_decimal("three", ["one", "four"])
        assert result == "three point one four"
    
    def test_format_percentage(self):
        result = self.config.format_percentage("fifty")
        assert result == "fifty percent"
    
    def test_format_ordinal_fallback(self):
        # English config uses fallback for ordinals
        assert self.config.format_ordinal(1, "st") == "1st"
        assert self.config.format_ordinal(2, "nd") == "2nd"
    
    def test_format_time_basic(self):
        result = self.config.format_time(1, 0, "one", "")
        assert result == "one o'clock"
        
        result = self.config.format_time(2, 0, "two", "")
        assert result == "two o'clock"
    
    def test_format_time_special_minutes(self):
        result = self.config.format_time(2, 15, "two", "fifteen")
        assert result == "quarter past two"
        
        result = self.config.format_time(2, 30, "two", "thirty")
        assert result == "half past two"
        
        result = self.config.format_time(2, 45, "two", "forty-five")
        assert result == "quarter to three"
    
    def test_format_time_regular_minutes(self):
        result = self.config.format_time(3, 25, "three", "twenty-five")
        assert result == "three twenty-five"
    
    def test_format_time_am_pm(self):
        result = self.config.format_time_am_pm(1, 0, "one", "", "AM")
        assert result == "one o'clock AM"
        
        result = self.config.format_time_am_pm(13, 0, "thirteen", "", "PM")
        assert result == "one o'clock PM"  # Converts to 12-hour
        
        result = self.config.format_time_am_pm(0, 0, "zero", "", "AM")
        assert result == "twelve o'clock AM"  # Midnight
    
    def test_format_time_am_pm_special_minutes(self):
        result = self.config.format_time_am_pm(14, 15, "fourteen", "fifteen", "PM")
        assert result == "quarter past two PM"
        
        result = self.config.format_time_am_pm(15, 45, "fifteen", "forty-five", "PM")
        assert result == "quarter to four PM"
    
    def test_convert_number_en_helper(self):
        # Test the private helper method
        assert self.config._convert_number_en(5) == "five"
        assert self.config._convert_number_en(15) == "fifteen"
        assert self.config._convert_number_en(25) == "twenty-five"
        assert self.config._convert_number_en(100) == "100"  # Fallback
    
    def test_format_hundreds_special_cases(self):
        # English has no special cases
        result = self.config.format_hundreds_special_cases(1, 0)
        assert result is None
    
    def test_format_scale_word(self):
        result = self.config.format_scale_word("two", "thousand", False)
        assert result == "two thousand"
        
        result = self.config.format_scale_word("one", "million", True)
        assert result == "one million"
    
    def test_get_words(self):
        assert self.config.get_zero_word() == "zero"
        assert self.config.get_negative_word() == "negative"