# Hafta 1 Notları

## Gün 0 doğrulamaları

- Yeni dal: git branch --show-current çıktısı ay4-paketleme.
- Sanal ortam: which python çıktısı /mnt/c/Users/hyild/Documents/microgpt-tr/.venv/bin/python.
- Paket iskeleti: microgpt_tr ve tests klasörleriyle microgpt_tr/__init__.py oluşturuldu.
- pyproject.toml proje köküne eklendi.
- Geliştirme paketleri: pytest 9.2.0, FastAPI 0.165.0, Uvicorn 0.54.0 ve HTTPX 0.28.1.
- tmux sürümü: 3.6.
- Duman testi: pytest -q sonucu 1 passed in 0.20s.
- .gitignore kontrolünde .venv/ ve .pytest_cache/ kurallarının çalıştığı görüldü.
- Gün 0 commit sonrası git status temizdi.

## Gün 1 Pazartesi-Salı

- Veri hazırlığı: 1.100 satırdan 1.086 benzersiz isim kaldı; 14 tekrar kaldırıldı.
- Seed 42 ile bölme: train 896, val 100, test 90.
- Tokenizer kontrolü: hüseyin, ömer ve süleyman encode/decode sonrasında aynı kaldı.
- Bigram kontrolü: BOS olasılıkları toplamı 1.0; train kümesinde 5.997 geçiş sayıldı.
