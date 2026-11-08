# rss-shield-highway

**Назначение.** RSS-фильтр безопасности для DQN-агента в highway-env: маска действий, задержка
исполнительного тракта τ = Δt, супервизорные команды TAKEOVER / ABORT / FALLBACK. Экспериментальный
протокол исследования — `docs/protocol/`.

**Текущее состояние.** Протокол исследования — `docs/protocol/Протокол_исследования_RSS-фильтр.docx`.
Пакет верификации: 87 тестов, все строки таблицы 3 — pass. Основное обучение DQN ещё не запускалось;
короткий сквозной пилот цепочки обучения — `experiments/pilot.py`, его отчёт CI публикует в PR.

## Быстрый старт

Требуется Python ≥ 3.10.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
PYTHON=python ./run_all.sh     # код выхода 0 = все строки таблицы 3 pass
```

`run_all.sh` прогоняет pytest, контрольные сценарии, профиль манёвра и собирает `results/` и `logs/`.
Сравнение с эталоном: `diff -r expected/ results/` (могут отличаться только время прогона,
строки ОС/Python и хэши, если менялся код).

## Карта репозитория

| Путь | Что это |
|---|---|
| `rss_shield/` | подкласс `MDPVehicle` (FIFO на 2 кадра, ограничители), фильтр с маской и иерархией, сценарии, интеграция с `HighwayEnv` |
| `tests/` | 87 pytest-тестов; `tests/reference.py` — независимый эталон таблицы 2 |
| `tools/make_table3.py` | сборка `results/table3.md` и `table3.csv` из журналов и pytest |
| `expected/` | эталонный прогон пакета верификации |
| `runs/verification_run/` | зафиксированный контрольный прогон (`results/`, `logs/`) |
| `sim/` | точное ускорение highway-env 1.12.1 и побитовый тест эквивалентности `sim/test_fast_road.py` |
| `docs/protocol/` | протокол исследования (`Протокол_исследования_RSS-фильтр.docx`) |
| `experiments/` | основной генератор трафика, 86-признаковое наблюдение, DQN на NumPy, semi-MDP-раннер со счётчиками, сквозной пилот |
| `configs/` | заготовки: сиды и сетка D × ρ |
| `.github/workflows/verify.yml` | CI: Python 3.13 и 3.14, сверка `table3.csv` с эталоном, тест эквивалентности `sim/` |
| `.github/workflows/pilot.yml` | CI: тесты `experiments/`, замер шага, короткий пилот; отчёт публикуется в PR |

## Правила

- `tests/reference.py` и `expected/` **не меняются**.
- Любое изменение `rss_shield/` — через PR с зелёным CI и обновлённым `expected/`; в описании PR
  указывается причина.
- `sim/` — точное ускорение highway-env для экспериментов; любое его изменение — только
  с зелёным `sim/test_fast_road.py`.
- Прямых коммитов в `main` нет: только через PR.
