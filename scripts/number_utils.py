import re

map = {
    "pt": {
        "0": "zero",
        "1": "um",
        "2": "dois",
        "3": "três",
        "4": "quatro",
        "5": "cinco",
        "6": "seis",
        "7": "sete",
        "8": "oito",
        "9": "nove",
        "10": "dez",
        "11": "onze",
        "12": "doze",
        "13": "treze",
        "14": "quatorze",
        "15": "quinze",
        "16": "dezesseis",
        "17": "dezessete",
        "18": "dezoito",
        "19": "dezenove",
        "20": "vinte",
    },
    "en": {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
        "11": "eleven",
        "12": "twelve",
        "13": "thirteen",
        "14": "fourteen",
        "15": "fifteen",
        "16": "sixteen",
        "17": "seventeen",
        "18": "eighteen",
        "19": "nineteen",
        "20": "twenty",
    },
}


def normalize_text_number(text: str, lan: str) -> str:
    """Normalize text for TTS training"""

    for num, word in map[lan].items():
        text = re.sub(rf"\b{num}\b", word, text)

    return " ".join(text.split())
