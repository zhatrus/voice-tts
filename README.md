# Ukrainian Voice (TTS) API

Self-hosted **український TTS-контейнер**: текст → аудіо. Один сучасний рушій
(**StyleTTS2 Ukrainian**, `patriotyk/styletts2_ukrainian_multispeaker`) дає і
готові голоси (пресети), і **клонування** власного голосу зі зразка. Орієнтований
на телефонію/IVR: підтримує вузькосмугові формати 8 кГц і μ-law (G.711),
нормалізацію українського тексту (числа, дати, абревіатури → словами) та наголоси.

Каркас навмисно повторює `audio-transcribe-api` (FastAPI, lazy-модель з idle-евікцією,
API-key, env-конфіг, GPU), щоб два сервіси були однакові в експлуатації.

## Можливості

- Синтез українською через StyleTTS2 (24 кГц), 31 пресетний голос (чол./жін.).
- **Клонування голосу** зі зразка аудіо (zero-shot, `predict_style_multi`).
- Телефонійні формати: `wav_8k`, `ulaw_8k`; також `wav`, `mp3`, `ogg`.
- Українська нормалізація чисел/дат/абревіатур (verbalizer) + наголоси.
- Кеш аудіо на диску (повторні IVR-репліки → миттєво).
- Опційна авторизація `X-API-Key`. OpenAI-сумісний ендпоінт `/v1/audio/speech`.

## Вимоги

- Docker + Docker Compose.
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  CPU технічно працює, але для телефонії надто повільний — GPU-only за замовчуванням.
- Перший запуск завантажує моделі (акустична + verbalizer) у том `models` — кілька хвилин.

## Швидкий старт

```bash
make env            # створити .env з шаблону
# за бажання відредагувати .env: API_KEY, DEFAULT_VOICE, DEFAULT_FORMAT

make build          # зібрати GPU-образ
make up             # запустити
make health         # -> {"status":"ok", ...}
make voices         # список голосів
```

## Конфігурація (`.env`)

| Змінна | Дефолт | Опис |
|---|---|---|
| `DEVICE` | `gpu` | `gpu` / `cpu` |
| `MODEL_REPO` | `patriotyk/styletts2_ukrainian_multispeaker` | акустична модель |
| `VOICES_SPACE_REPO` | `patriotyk/styletts2-ukrainian` | джерело пресетних стилів `voices/*.pt` |
| `DEFAULT_VOICE` | (перший за абеткою) | голос за замовчуванням |
| `DEFAULT_FORMAT` | `wav` | `wav` / `wav_8k` / `ulaw_8k` / `mp3` / `ogg` |
| `DEFAULT_SAMPLE_RATE` | `24000` | для 8k-форматів примусово 8000 |
| `VERBALIZE` | `true` | розгортати числа/дати/абревіатури словами |
| `MAX_TEXT_LEN` | `3000` | ліміт довжини тексту |
| `MODEL_IDLE_TIMEOUT_MIN` | `30` | вивантажити модель з VRAM після N хв простою (`0` = ніколи) |
| `CACHE_ENABLED` | `true` | кешувати згенероване аудіо на диск |
| `API_KEY` | — | якщо задано, потрібен заголовок `X-API-Key` |
| `HF_TOKEN` | — | опційно, підвищує ліміти завантаження з HF |
| `PORT` | `8001` | порт на хості (контейнер завжди слухає 8000) |

## API

Усі ендпоінти, крім `/health`, вимагають `X-API-Key`, **якщо** `API_KEY` заданий.

### `POST /tts`
```bash
curl -X POST http://localhost:8001/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Вітаю! Ваш баланс — 25 гривень 50 копійок.","voice":"Тетяна Гончарова","format":"ulaw_8k"}' \
  --output speech.wav
```
Поля: `text` (обов'язкове), `voice`, `speed` (дефолт 1.0), `format`, `sample_rate`.
Наголос можна задати вручну знаком `+` перед голосною: `за+мок` / `замо+к`.

### `GET /voices`
```json
{ "presets": ["Артем Окороков", "Тетяна Гончарова", ...], "clones": [], "count": 31 }
```

### `POST /voices/clone` (multipart/form-data)
Створити голос зі зразка (рекомендовано чистий запис 10–30 с, моно):
```bash
curl -X POST http://localhost:8001/voices/clone \
  -F "name=Мій голос" \
  -F "file=@sample.wav"
# -> {"voice_id": "Мій_голос"}
```
Далі використовувати як звичайний голос: `{"text":"...","voice":"Мій_голос"}`.

### `POST /v1/audio/speech` (OpenAI-сумісний)
```bash
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Привіт від голосового бота.","voice":"Артем Окороков","response_format":"mp3"}' \
  --output speech.mp3
```

## Телефонія / IVR

- Формат `ulaw_8k` — класичний G.711 μ-law 8 кГц для телефонії; `wav_8k` — PCM16 8 кГц.
- `VERBALIZE=true` критичний: інакше «25,50 грн», номери й дати читатимуться неправильно.
- Сталі репліки IVR кешуються (`CACHE_ENABLED`) — повтор віддається з диска без синтезу.

## n8n

`HTTP Request` → `POST /tts` (JSON, `Response Format: File`) → отримане аудіо
передати далі (відтворення в дзвінку/збереження). Для OpenAI-нод використовуйте
`/v1/audio/speech` із базовим URL цього сервісу.

## Розробка / збірка локально (без Docker)

Залежності важкі й GPU-орієнтовані — рекомендована робота лише в контейнері на
сервері. Локально на Windows повноцінний запуск не передбачений.

## Перевірка на сервері (перший запуск)

Цей контейнер призначений для вашого GPU-сервера (як audio-transcribe-api).
Що звірити після першого `make up`:

1. `make logs` — модель та стилі завантажились без помилок (`Model preloaded`).
2. `make voices` — повертає список ~31 голосу.
3. `POST /tts` з коротким текстом — приходить валідний аудіофайл.
4. Якщо verbalizer не вантажиться — він деградує м'яко (текст іде як є);
   за потреби вимкніть `VERBALIZE=false`.
5. Клонування (`predict_style_multi`) перевірити окремо на реальному зразку —
   семантику стилю варто підтвердити на живій моделі.

## Ліцензії / походження

- Рушій і моделі: [patriotyk/styletts2-ukrainian](https://huggingface.co/spaces/patriotyk/styletts2-ukrainian),
  [styletts2-inference](https://github.com/patriotyk/styletts2-inference) (MIT).
- Front-end: [ukrainian-word-stress](https://github.com/patriotyk/ukrainian-word-stress),
  [ipa-uk](https://github.com/patriotyk/ipa-uk).
- Verbalizer: `skypro1111/m2m100-ukr-verbalization-ct2`.
