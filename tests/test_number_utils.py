import pytest
from scripts.number_utils import normalize_text_number


class TestNormalizeTextNumber:
    def test_portuguese_basic_numbers(self):
        assert normalize_text_number("1 gato", "pt") == "um gato"
        assert normalize_text_number("2 cachorros", "pt") == "dois cachorros"
        assert normalize_text_number("10 pessoas", "pt") == "dez pessoas"
        
    def test_portuguese_teens(self):
        assert normalize_text_number("11 anos", "pt") == "onze anos"
        assert normalize_text_number("15 minutos", "pt") == "quinze minutos"
        assert normalize_text_number("19 dias", "pt") == "dezenove dias"
        
    def test_portuguese_twenty(self):
        assert normalize_text_number("20 metros", "pt") == "vinte metros"
        
    def test_english_basic_numbers(self):
        assert normalize_text_number("1 cat", "en") == "one cat"
        assert normalize_text_number("2 dogs", "en") == "two dogs"
        assert normalize_text_number("10 people", "en") == "ten people"
        
    def test_english_teens(self):
        assert normalize_text_number("11 years", "en") == "eleven years"
        assert normalize_text_number("15 minutes", "en") == "fifteen minutes"
        assert normalize_text_number("19 days", "en") == "nineteen days"
        
    def test_english_twenty(self):
        assert normalize_text_number("20 meters", "en") == "twenty meters"
        
    def test_word_boundaries(self):
        # Should only replace whole numbers, not parts of words
        assert normalize_text_number("12345", "pt") == "12345"  # Large number not in map
        assert normalize_text_number("a1b", "pt") == "a1b"  # Not word boundary
        
    def test_multiple_numbers(self):
        assert normalize_text_number("1 e 2", "pt") == "um e dois"
        assert normalize_text_number("1 and 2", "en") == "one and two"
        
    def test_whitespace_normalization(self):
        assert normalize_text_number("1  2   3", "pt") == "um dois três"
        assert normalize_text_number(" 1 ", "en") == "one"
        
    def test_unsupported_language_raises_error(self):
        # Function raises KeyError for unsupported languages
        with pytest.raises(KeyError):
            normalize_text_number("1 test", "fr")

    def test_zero_handling(self):
        assert normalize_text_number("0 items", "pt") == "zero items"
        assert normalize_text_number("0 items", "en") == "zero items"