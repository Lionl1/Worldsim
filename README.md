# 🌌 worldsim

**Открытая модель вселенной на Rust: от броска камня до космической паутины. Каждый закон проверяется автоматически.**

*An open, dependency-free Rust model of the universe — planets, climate, star systems, galaxies and cosmology —
with an in-browser 3D viewer, real star/exoplanet catalogs, dataset generation and 59 automated physics checks.*

```bash
git clone https://github.com/Lionl1/worldsim && cd worldsim
cargo run --release          # соберёт и откроет 3D-интерфейс в браузере (http://localhost:8080)
```

Нужен только Rust ≥ 1.75. Внешних крейтов нет. Браузеру нужен интернет только для загрузки three.js с CDN.

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
| ✅ **Законы** | 59 тестов | см. ниже |

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

## Командная строка

```bash
worldsim info --physics mars --atmosphere earth --surface moon   # смешанный мир
worldsim throw mars --speed 100 --elev 45 --out traj.csv
worldsim thermal moon --lat 0
worldsim particle earth --species proton --mev 1 --L 3 --pitch 40
worldsim system --bodies sun,earth,moon,mars,jupiter --years 10 --gr
worldsim dataset --task projectile --n 10000 --worlds earth,moon,mars,mars:earth:moon --noise 0.01 --out data/proj
worldsim serve --port 8080
```

Задачи датасетов: `projectile`, `orbit`, `thermal`, `particle`, `system`. У каждого сэмпла есть
`validation_error` и `valid`, а `meta.json` хранит миры, seed и константы.

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
  validate                                    — 59 проверок законов
  datagen  server  json  main                 — датасеты, HTTP-сервер, CLI
web/index.html                                — 3D-интерфейс (three.js), встроен в бинарник
data/                                         — каталоги HYG и OEC (свои лицензии)
tools/build_data.py                           — пересборка данных из каталогов
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
