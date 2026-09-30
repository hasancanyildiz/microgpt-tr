# Gün 1: Step 0.1 (Veri seti)

##0.1 The Dataset
- open('input.txt').read() : Bütün dosyayı tek bir strng olarak okur
- .strip() :Boşlukları siler
- .split('\n') : Satırları listeye böl
- l.strip(): Boi satırları listeye almaz
- random.shuffle(docs) : İsimlerin sırasını rasgele değiştirir.
## Bizim projede fark
- Kendi verimiz Türkçe isimler, ~1.000 tane
- Vocab'a ç ğ ı ö ş ü girecek
