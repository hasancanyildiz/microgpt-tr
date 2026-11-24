"""İsim verisini yükleme, temizleme ve bölme işlemlerini içerir.
Türkçe harfleri normalleştirir, tekrarları kaldırır ve isimleri kümelere ayırır.
"""

import random


def load_names(path):
    names = []
    seen = set()

    with open(path, encoding="utf-8") as file:
        for line in file:
            name = (
                line.strip()
                .replace(" ", "")
                .replace("İ", "i")
                .replace("I", "ı")
                .lower()
            )

            if name and name not in seen:
                seen.add(name)
                names.append(name)

    return names


def split_names(names, seed=42):
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)

    val_names = shuffled[:100]
    test_names = shuffled[100:190]
    train_names = shuffled[190:]

    return train_names, val_names, test_names

class Tokenizer:
    """İsimleri karakter kimliklerine dönüştürür ve geri çözer."""

    def __init__(self, names):
        self.chars = sorted(set("".join(names)))
        self.char_to_id = {
            char: index for index, char in enumerate(self.chars)
        }
        self.id_to_char = {
            index: char for char, index in self.char_to_id.items()
        }
        self.BOS = len(self.chars)
        self.vocab_size = self.BOS + 1

    def encode(self, name):
        return [self.BOS] + [
            self.char_to_id[char] for char in name
        ] + [self.BOS]

    def decode(self, tokens):
        return "".join(
            self.id_to_char[token]
            for token in tokens
            if token != self.BOS
        )
