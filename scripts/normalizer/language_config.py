import re
from abc import ABC, abstractmethod
from typing import List, Tuple


class LanguageConfig(ABC):
    """Abstract base class for language-specific normalization configurations."""

    @property
    @abstractmethod
    def units(self) -> List[str]:
        """Returns a list of units for the language."""
        pass

    @property
    @abstractmethod
    def tens(self) -> List[str]:
        """Returns a list of tens for the language."""
        pass

    @property
    @abstractmethod
    def hundreds(self) -> List[str]:
        """Returns a list of hundreds for the language."""
        pass

    @property
    @abstractmethod
    def scales(self) -> List[Tuple[str, str]]:
        pass

    @property
    @abstractmethod
    def decimal_separator(self) -> str:
        pass

    @property
    @abstractmethod
    def thousands_separator(self) -> str:
        pass

    @property
    @abstractmethod
    def connector_word(self) -> str:
        pass

    @abstractmethod
    def format_decimal(self, whole_words: str, decimal_digits: List[str]) -> str:
        pass

    @abstractmethod
    def format_percentage(self, number_words: str) -> str:
        pass

    @abstractmethod
    def format_ordinal(self, num: int, suffix: str) -> str:
        pass

    @abstractmethod
    def format_time(
        self, hours: int, minutes: int, hour_words: str, minute_words: str
    ) -> str:
        pass

    @abstractmethod
    def format_hundreds_special_cases(
        self, hundreds: int, remainder: int
    ) -> str | None:
        pass

    @abstractmethod
    def format_time_am_pm(
        self, hours: int, minutes: int, hour_words: str, minute_words: str, period: str
    ) -> str:
        pass

    @abstractmethod
    def format_scale_word(
        self, quotient_words: str, scale_word: str, is_singular: bool
    ) -> str:
        pass

    @abstractmethod
    def get_zero_word(self) -> str:
        pass

    @abstractmethod
    def get_negative_word(self) -> str:
        pass
