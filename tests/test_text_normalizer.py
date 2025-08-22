import pytest
from scripts.normalizer.text_normalizer import TextNormalizer
from scripts.normalizer.port_config import PortugueseConfig
from scripts.normalizer.en_config import EnglishConfig


class TestTextNormalizer:
    def setup_method(self):
        self.normalizer = TextNormalizer()
    
    def test_initialization(self):
        assert 'pt' in self.normalizer.languages
        assert 'en' in self.normalizer.languages
        assert isinstance(self.normalizer.languages['pt'], PortugueseConfig)
        assert isinstance(self.normalizer.languages['en'], EnglishConfig)
    
    def test_register_language(self):
        mock_config = PortugueseConfig()
        self.normalizer.register_language('es', mock_config)
        assert 'es' in self.normalizer.languages
    
    def test_number_to_words_basic(self):
        # Portuguese
        assert self.normalizer.number_to_words(0, 'pt') == 'zero'
        assert self.normalizer.number_to_words(1, 'pt') == 'um'
        assert self.normalizer.number_to_words(15, 'pt') == 'quinze'
        assert self.normalizer.number_to_words(20, 'pt') == 'vinte'
        
        # English
        assert self.normalizer.number_to_words(0, 'en') == 'zero'
        assert self.normalizer.number_to_words(1, 'en') == 'one'
        assert self.normalizer.number_to_words(15, 'en') == 'fifteen'
        assert self.normalizer.number_to_words(20, 'en') == 'twenty'
    
    def test_number_to_words_tens(self):
        # Portuguese
        assert self.normalizer.number_to_words(21, 'pt') == 'vinte e um'
        assert self.normalizer.number_to_words(35, 'pt') == 'trinta e cinco'
        assert self.normalizer.number_to_words(99, 'pt') == 'noventa e nove'
        
        # English - note: English doesn't use connector word
        assert self.normalizer.number_to_words(21, 'en') == 'twenty-one'
        assert self.normalizer.number_to_words(35, 'en') == 'thirty-five'
    
    def test_number_to_words_hundreds(self):
        # Portuguese special case for 100
        assert self.normalizer.number_to_words(100, 'pt') == 'cem'
        assert self.normalizer.number_to_words(101, 'pt') == 'cento e um'
        assert self.normalizer.number_to_words(200, 'pt') == 'duzentos'
        
        # English
        assert self.normalizer.number_to_words(100, 'en') == 'one hundred'
        assert self.normalizer.number_to_words(101, 'en') == 'one hundred one'
        assert self.normalizer.number_to_words(200, 'en') == 'two hundred'
    
    def test_number_to_words_thousands(self):
        # Portuguese
        assert self.normalizer.number_to_words(1000, 'pt') == 'mil'
        assert self.normalizer.number_to_words(2000, 'pt') == 'dois mil'
        assert self.normalizer.number_to_words(1001, 'pt') == 'mil e um'
        
        # English
        assert self.normalizer.number_to_words(1000, 'en') == 'one thousand'
        assert self.normalizer.number_to_words(2000, 'en') == 'two thousand'
    
    def test_number_to_words_negative(self):
        assert self.normalizer.number_to_words(-5, 'pt') == 'menos cinco'
        assert self.normalizer.number_to_words(-100, 'pt') == 'menos cem'
        
        assert self.normalizer.number_to_words(-5, 'en') == 'negative five'
        assert self.normalizer.number_to_words(-100, 'en') == 'negative one hundred'
    
    def test_normalize_text_integers(self):
        # Portuguese
        text_pt = "Eu tenho 25 anos e 3 gatos"
        expected_pt = "Eu tenho vinte e cinco anos e três gatos"
        assert self.normalizer.normalize_text(text_pt, 'pt') == expected_pt
        
        # English
        text_en = "I have 25 years and 3 cats"
        expected_en = "I have twenty-five years and three cats"
        assert self.normalizer.normalize_text(text_en, 'en') == expected_en
    
    def test_normalize_text_time_format(self):
        # Basic time format - Portuguese uses "dois" (masculine) for hours, not "duas" (feminine)
        assert "dois horas e quinze" in self.normalizer.normalize_text("2:15", 'pt')
        
        # AM/PM format - English converts 2:30 PM to "half past two"
        result = self.normalizer.normalize_text("2:30 PM", 'en')
        assert "half past two" in result.lower()
    
    def test_normalize_text_percentages(self):
        # Portuguese
        result = self.normalizer.normalize_text("50%", 'pt')
        assert "cinquenta por cento" in result
        
        # English
        result = self.normalizer.normalize_text("25%", 'en')
        assert "twenty-five percent" in result
    
    def test_normalize_text_currency_portuguese(self):
        result = self.normalizer.normalize_text("R$ 1.250,50", 'pt')
        assert "reais" in result and "centavos" in result
    
    def test_normalize_text_currency_english(self):
        result = self.normalizer.normalize_text("$1,250.50", 'en')
        assert "dollars" in result and "cents" in result
    
    def test_normalize_text_decimals(self):
        # Portuguese uses comma as decimal separator
        result = self.normalizer.normalize_text("3,14", 'pt')
        assert "três vírgula um quatro" in result
        
        # English uses period as decimal separator
        result = self.normalizer.normalize_text("3.14", 'en')
        assert "three point one four" in result
    
    def test_normalize_text_thousands(self):
        # Portuguese uses period as thousands separator
        result = self.normalizer.normalize_text("1.000", 'pt')
        assert "mil" in result
        
        # English uses comma as thousands separator
        result = self.normalizer.normalize_text("1,000", 'en')
        assert "one thousand" in result
    
    def test_normalize_text_ordinals_portuguese(self):
        result = self.normalizer.normalize_text("1º lugar", 'pt')
        assert "primeiro" in result
        
        result = self.normalizer.normalize_text("2ª vez", 'pt')
        assert "segunda" in result
    
    def test_normalize_text_whitespace_cleanup(self):
        text = "  1   2   3  "
        result = self.normalizer.normalize_text(text, 'pt')
        # Should normalize whitespace and convert numbers
        assert result.strip() == "um dois três"
    
    def test_normalize_text_unsupported_language(self):
        with pytest.raises(ValueError, match="Language 'fr' not supported"):
            self.normalizer.normalize_text("Hello 123", 'fr')
    
    def test_normalize_text_complex_mixed_content(self):
        # Test with multiple types of numeric content
        text = "At 2:30 PM, the price was $1,250.50, which is about 75% of the original value."
        result = self.normalizer.normalize_text(text, 'en')
        
        # Should convert all numeric elements
        assert "two" in result.lower()
        assert "half past" in result.lower()  # 2:30 PM becomes "half past two PM"
        assert "dollars" in result.lower()
        assert "cents" in result.lower()
        assert "percent" in result.lower()