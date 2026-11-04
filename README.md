# 🌌 worldsim

**Открытая физика на Rust: от анализа космической миссии до проверки обученных world models. Каждый закон проверяется автоматически.**

*Open, dependency-free physics in Rust: Mission Lab turns mission requirements into a scorecard and technology trade study, and a qualification ladder tells you when a learned world model can be trusted. Includes robot environments, counterfactual data, a 3D viewer from planets to cosmology, and 73 automated physics checks.*

**Для инженеров миссий:** [Mission Lab](#mission-lab) отвечает, выполняются ли требования, что ломает миссию и какая технология окупится. **Для разработчиков моделей:** [сбор данных и квалификация](#world-models-данные-и-квалификация) показывают, сколько секунд можно верить модели и видит ли она аварии заранее. Что это умеет, чего не умеет и на каких исследованиях основано, описано в [docs/PRODUCT.md](docs/PRODUCT.md).

```bash
git clone https://github.com/Lionl1/worldsim && cd worldsim
./tools/fetch_data.sh        # один раз: скачать каталоги звёзд и экзопланет (нужны git и python3)
cargo run --release          # соберёт и откроет 3D-интерфейс в браузере (http://localhost:8080)
```

Нужен только Rust ≥ 1.75 — внешних крейтов нет. Каталоги (вкладка «Карта») скачиваются скриптом из открытых репозиториев. Браузеру нужен интернет только для загрузки three.js с CDN.

## Mission Lab

```bash
cargo run --release -- mission missions/lunar_lander.json --out report
```

Описание миссии задаёт аппарат, разбросы, требования и варианты технологий. Готовые примеры лежат в [missions/](missions/). Результат есть и на вкладке «Mission Lab» в интерфейсе.

| Вопрос | Что получите |
|---|---|
| Выполнима ли миссия? | Карта требований с интервалом 95%: выполнено, не выполнено или данных мало |
| Как отказывает? | Виды отказов: большая скорость касания, опрокидывание и другие |
| Что её ломает? | Влияние каждого разброса в отдельности |
| Что купить? | Парное сравнение технологий на тех же условиях: что спасено, что потеряно, сколько это весит |

Пример ответа для лунного модуля: успех 90,7%, требований выполнено 0 из 4, а двигатель 7 кН даёт +4,3 ± 2,8 п.п. за 25 кг и закрывает два требования.

Физика упрощена: это инструмент для ранних проектных решений, а не замена высокоточному анализу и испытаниям.

## Что внутри

| Вкладка | Модель | Законы |
|---|---|---|
| 🪐 **Миры** | Земля, Луна, Марс, Венера, Меркурий, Юпитер, Солнце + любые смеси (физика ≠ атмосфера ≠ кора) | тяготение, J2, Кориолис, гидростатика, идеальный газ, Архимед, сопротивление, стехиометрия |
| 🌡️ **Климат** | сезонная широтная модель энергобаланса для любого мира | инсоляция из орбиты, Стефан–Больцман, парниковая атмосфера Эддингтона, лёд–альбедо, 1-е начало |
| ☀️ **Солнечная система** | N тел по элементам J2000 | Ньютон, симплектический интегратор Йошиды, ОТО (1PN) |
| ✨ **Звёздные системы** | генератор: звезда по IMF Крупы, планеты по устойчивости Хилла | масса–светимость, Планк → цвет, обитаемая зона Копарапу, масса–радиус Чена–Киппинга |
| 🔭 **Карта** | 11 270 звёзд HYG + 5 283 экзопланеты | любую экзопланету — в модель мира одним кликом |
| 🌀 **Галактики** | Млечный Путь, столкновение галактик, скопления | Барнс–Хат O(N log N), живое тёмное гало, вириал, кривая вращения |
| 🕸️ **Космология** | ΛCDM, рост структур из Большого взрыва | Фридман, начальные условия Зельдовича, спектр BBKS/σ₈, particle-mesh через FFT |
| 🚀 **Mission Lab** | анализ миссии по требованиям | Монте-Карло, интервалы Уилсона, парные сравнения |
| 🤖 **Роботы** | посадочный модуль, дрон, ровер в реальном времени | Циолковский, импульсная теория винта, трение Кулона, контакт опор, устойчивость |
| ✅ **Законы** | 73 теста | см. ниже |

## Проверка законов

```bash
cargo run --release -- validate     # таблица, ~2–3 с
cargo test --release                # то же, как тесты
```

Примеры того, что проверяется, и что получается:

| Закон | Тест | Результат |
|---|---|---|
| Сохранение энергии | Солнечная система, 1 год | ΔE/E ≈ 2·10⁻¹³ |
| ОТО | прецессия перигелия Меркурия | 42.7″/век (теория 43.0″) |
| Кориолис | падение со 100 м на экваторе | 21.977 мм (теория 21.977) |
| Гидростатика | давление Земли на 11 км | 22 646 Па (US Std 22 632) |
| Тепло + излучение | Луна, экватор | 98–386 К (Diviner ≈ 95–390) |
| Климат | Земля: средняя T / кромка льда | 289.6 К / 75° |
| Бистабильность | холодный старт Земли | «Земля-снежок» |
| Вириал | сфера Пламмера | 2K/|W| = 1.006 |
| Кривая вращения | Млечный Путь на 8 кпк | 212 км/с |
| Фридман | возраст Вселенной (Planck 2018) | 13.786 млрд лет |
| Рост структур | P(k) в PM против D(a)² | в пределах 4% |
| Планк → Стефан–Больцман, Вин | численное интегрирование спектра | 10⁻⁸ / 10⁻⁶ |
| Уравнение Циолковского | посадочный модуль, полный расход топлива | Δv = 1093.18 м/с (теория 1093.14) |
| Импульсная теория винта | дрон на Марсе: обычные винты / площадь ×40 | −3.41 / +8.88 м/с (падает / взлетает) |
| Контакт опор | модуль стоит на Марсе | акселерометр показывает g, ±3% |
| Устойчивость v·ω/g | тот же поворот ровера | на Луне опрокидывает, на Земле нет |
| Точность ветвления | копия среды против оригинала | совпадают до 10⁻¹² |

## Командная строка

```bash
worldsim info --physics mars --atmosphere earth --surface moon   # смешанный мир
worldsim throw mars --speed 100 --elev 45 --out traj.csv
worldsim thermal moon --lat 0
worldsim particle earth --species proton --mev 1 --L 3 --pitch 40
worldsim system --bodies sun,earth,moon,mars,jupiter --years 10 --gr
worldsim dataset --task projectile --n 10000 --worlds earth,moon,mars,mars:earth:moon --noise 0.01 --out data/proj
worldsim mission missions/lunar_lander.json --out report
worldsim collect --task lander --worlds moon,mars:earth:moon --episodes 1000 --randomize --counterfactuals 3 --out runs/data
worldsim serve --port 8080
```

Задачи датасетов: `projectile`, `orbit`, `thermal`, `particle`, `system`. У каждого сэмпла есть
`validation_error` и `valid`, а `meta.json` хранит миры, seed и константы.

## Среда для обучения роботов

Три задачи в стиле Gymnasium. Агент видит только зашумлённые показания сенсоров и выдаёт непрерывные команды приводам,
а физика та же, что в остальном worldsim, поэтому одного агента можно учить на разных и смешанных мирах.

| Задача | Цель | Действия | Наблюдения |
|---|---|---|---|
| `lander` | мягко сесть на площадку | `throttle` 0…1, `torque_x/y/z` −1…1 | 35: GPS, скорость, ориентация, гироскоп, радиовысотомер, акселерометр, топливо, давление, лидар, признаки попадания лучей |
| `drone` | долететь до точки и зависнуть | `rotor_1…4` 0…1 | 20: цель, скорость, ориентация, гироскоп, радиовысотомер, акселерометр, батарея, давление, плотность воздуха, признак попадания |
| `rover` | доехать до цели по рельефу | `left`, `right` −1…1 | 44: цель, курс, скорость, наклон, батарея, высота Солнца, температура, давление, лидар, признаки попадания |

Модуль стоит на четырёх упругих опорах, поэтому модель видит сам контакт. Дрон стартует в воздухе и может сесть и снова взлететь. Ровер опрокидывается, если уклон и инерция поворота превышают запас устойчивости.

Сложность добавляют порывы ветра (`gust`), задержка приводов (`lag`), скрытый отказ привода (`fault`) и смена мира между эпизодами (`worlds`).

### HTTP API

```bash
worldsim serve --port 8080 --no-open

curl 'http://localhost:8080/api/env/make?task=lander&physics=moon&seed=42'   # вернёт id сессии, имена и границы
curl 'http://localhost:8080/api/env/step?id=1&action=0.5,0,0,0'              # одно действие
curl 'http://localhost:8080/api/env/step?id=1&mode=auto&repeat=100&text=1'   # 100 шагов автопилота + описание текстом
```

`step` возвращает `obs`, `reward`, `terminated`, `truncated` и `info`. Сервер держит соединение открытым (keep-alive), поэтому
агент может слать много запросов подряд через одно подключение. Числа вне допустимого диапазона дают ошибку 400.

Сервер хранит до 1024 сред и 64 тяжёлые симуляции. Когда места нет, удаляется давно не использованная сессия того же рода. Закончив работу, вызывайте `close`.

| Запрос `/api/env/…` | Параметры |
|---|---|
| `tasks`, `list` | — |
| `make` | `task`, `physics`, `worlds`, `atmosphere`, `surface`, `gravity_scale`, `randomize=1`, `noise` (0 — идеальные сенсоры), `rotor_scale`, `max_steps`, `frame_skip`, `gust`, `lag`, `fault`, `seed` |
| `reset` | `id`, `seed` |
| `step` | `id`, `action=a,b,…` или `mode=auto\|random`, `repeat` (1–100), `text=1`, `traj=1` (все промежуточные шаги) |
| `state` | `id`: также истинное состояние и скрытые параметры |
| `fork` | `id`: копия среды, чтобы проиграть другое действие из той же точки |
| `close` | `id` |
| `terrain` | `id`, `n` (сетка 17–257) |
| `camera` | `id`, `w`, `h` (4–256), `fov`, `mode=depth\|rgb` |

### Из Rust

```rust
use worldsim::env::{Env, EnvConfig};
let mut cfg = EnvConfig::new("lander", "moon")?;
cfg.randomize = true;                  // случайные параметры в каждом эпизоде
let mut env = Env::new(cfg)?;
env.reset(42);
loop {
    let action = env.autopilot();      // замените своей политикой: наблюдения → действие
    let (_obs, _reward, terminated, truncated) = env.step(&action);
    if terminated || truncated { break; }
}
println!("{}: {:.1}", env.info.reason, env.ep_return);
```

### Базовая линия

С этими числами можно сравнивать обученного агента. Встроенный ПД-автопилот, успешные эпизоды из 10, со случайными
параметрами и стандартным размером винтов:

| Задача | Земля | Луна | Марс |
|---|---|---|---|
| `lander` | 10 | 10 | 10 |
| `drone` | 10 | 0 | 0 |
| `rover` | 8 | 6 | 4 |

На Луне дрону не на что опереться: там нет воздуха. На Марсе обычным винтам не хватает плотности атмосферы. Повторить замер:
`cargo run --release --example probe_env`.

## World models: данные и квалификация

```bash
cargo build --release
worldsim collect --task lander --worlds moon --episodes 500 --randomize --noise 0 --counterfactuals 3 --out runs/moon
worldsim collect --task lander --worlds earth --episodes 500 --randomize --noise 0 --counterfactuals 3 --out runs/earth
worldsim collect --task lander --worlds mars:earth:moon --episodes 300 --randomize --noise 0 --counterfactuals 3 --out runs/mix

cd python && pip install -e '.[baseline]'
python -m worldsim.baseline train --data ../runs/moon ../runs/earth --out ../runs/base.pt
python -m worldsim.baseline qualify --model ../runs/base.pt --world moon=../runs/moon --world mix=../runs/mix --out ../runs/report
```

`collect` пишет `.npy` и `meta.json`: наблюдения, действия, истинное состояние, скрытые параметры и причины окончания эпизодов. Результат не зависит от числа потоков. С `--counterfactuals` добавляются ветвления: разные действия из одной точки с истинным ответом.

Подключить свою модель можно любой архитектуры:

```python
from worldsim import Dataset, WorldModel, qualify, to_markdown

class MyModel(WorldModel):
    name = "my-rssm"
    def init(self, obs, params):      # начальное состояние по наблюдениям
        return encode(obs)
    def step(self, state, action):    # шаг в воображении
        state = dynamics(state, action)
        return state, decode(state)   # предсказание в каналах сенсоров

report = qualify(MyModel(), {"moon": Dataset("runs/moon"), "mix": Dataset("runs/mix")})
print(to_markdown(report))
```

| Ступень | Что проверяет |
|---|---|
| L1 Точность | горизонт доверия при допусках миссии |
| L2 Физика | нарушения законов в воображении относительно симулятора |
| L3 Отклик | эффект действия против истинных ветвлений |
| L4 Оптимизм | пропущенные аварии и ложные тревоги |
| L5 Границы | те же проверки в мирах вне обучения |

Для обучения без файлов есть векторная среда, которая работает без HTTP:

```python
from worldsim import VectorEnv
env = VectorEnv("lander", "moon,mars,mars:earth:moon", n=64, randomize=True, gust=2.0, threads=8)
obs = env.reset()
obs, reward, terminated, truncated, info = env.step(env.autopilot())
```

## Как библиотека

```rust
use worldsim::{world::World, climate::{simulate, ClimateOptions}};
let mars = World::preset("mars")?;
let c = simulate(&mars, &ClimateOptions::default());
println!("Марс: {:.0} К, CO₂-шапки от {:.0}°", c.global_mean, c.ice_edge);
```

## Устройство

```
src/
  constants  presets  world  atmosphere      — константы CODATA, пресеты, сборка мира
  ballistics  thermal  magneto  nbody  orbit  — механика, тепло, частицы, N тел
  stars  starsystem                           — звёзды, генератор систем
  climate                                     — сезонная модель энергобаланса
  galaxy                                      — Барнс–Хат, начальные условия галактик
  cosmology                                   — Фридман, FFT, particle-mesh
  env                                         — среда для обучения роботов: lander, drone, rover
  validate                                    — 73 проверки законов
  mission  collect  ffi                       — Mission Lab, сбор переходов, векторная среда для Python
  datagen  server  json  main                 — датасеты, HTTP-сервер, CLI
web/index.html                                — 3D-интерфейс (three.js), встроен в бинарник
missions/                                     — примеры миссий
python/worldsim/                              — клиент, векторная среда, данные, квалификация, базовая модель
docs/PRODUCT.md                               — продукт, исследования, ограничения, планы
data/                                         — каталоги HYG и OEC (скачиваются tools/fetch_data.sh, свои лицензии)
tools/fetch_data.sh, build_data.py           — загрузка и сборка данных из каталогов
```

## Сравнение с аналогами

Universe Sandbox, SpaceEngine, OpenSpace, REBOUND, GADGET-4/SWIFT, ExoPlaSim — в [COMPARISON.md](COMPARISON.md):
где worldsim сильнее, где уступает и что в плане.

## Границы модели

Классическая механика, термодинамика, электромагнетизм, СТО и 1PN-поправка ОТО. Не моделируются:
квантовые эффекты, сильные гравитационные поля, гидродинамика (ветры, облака, газ в галактиках),
тектоника, химические реакции и разрушающие столкновения. Пресеты — средние значения из открытых
источников (NASA Fact Sheets, JPL, US Std Atmosphere 1976, Planck 2018 и др.).

## Лицензия

Код — MIT. Данные в `data/` — под своими лицензиями: HYG — CC BY-SA 4.0, Open Exoplanet Catalogue — MIT.
