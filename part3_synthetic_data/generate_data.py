import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

START_DATE = "2023-01-01"
OUTLIER_RATE = 0.015      # доля выбросов (>= 1%)
MISSING_RATE = 0.008      # доля пропусков в каждом из полей (>= 0.5%)
MIN_STORES, MAX_STORES = 5, 10
MAX_YEARS = 10
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "sales.csv")

# (город, коэффициент уровня цен)
CITIES = [("Москва", 1.12), ("Санкт-Петербург", 1.08), ("Казань", 0.98),
          ("Новосибирск", 0.95), ("Екатеринбург", 1.00), ("Самара", 0.94),
          ("Нижний Новгород", 0.96), ("Краснодар", 1.02), ("Омск", 0.90),
          ("Пермь", 0.92)]


class GeneratorError(Exception):
    """базовое исключение генератора."""


class InvalidConfigError(GeneratorError):
    """некорректные параметры запуска."""


class OutputError(GeneratorError):
    """ошибка записи результата на диск."""


@dataclass(frozen=True)
class Product:
    """описание товара и закономерностей его продаж."""
    name: str
    category: str
    base_qty: float        # средний дневной спрос в магазине
    base_price: float
    cost_ratio: float      # себестоимость как доля цены
    trend: float           # годовой рост (+) или спад (-) спроса
    weekend: float         # множитель спроса в выходные
    season_amp: float      # амплитуда годовой сезонности
    peak_day: int          # день года максимума сезонности
    promo_lift: float      # множитель спроса при акции
    promo_discount: float  # множитель цены при акции


PRODUCTS = [
    Product("Хлеб", "Продукты", 60, 45, 0.70, -0.03, 1.15, 0.03, 15, 1.30, 0.90),
    Product("Молоко", "Продукты", 50, 90, 0.72, 0.02, 1.10, 0.04, 20, 1.35, 0.88),
    Product("Яблоки", "Продукты", 40, 120, 0.65, 0.05, 1.25, 0.30, 270, 1.50, 0.85),
    Product("Вода", "Напитки", 80, 40, 0.55, 0.06, 1.20, 0.35, 200, 1.40, 0.90),
    Product("Сок", "Напитки", 45, 110, 0.62, -0.04, 1.25, 0.25, 200, 1.50, 0.85),
    Product("Кофе", "Напитки", 30, 450, 0.68, 0.10, 1.05, 0.12, 15, 1.40, 0.88),
    Product("Мыло", "Бытовая химия", 25, 70, 0.60, 0.00, 0.85, 0.05, 1, 1.30, 0.90),
    Product("Порошок", "Бытовая химия", 20, 380, 0.70, -0.08, 0.90, 0.04, 100, 1.60, 0.82),
    Product("Салфетки", "Бытовая химия", 35, 60, 0.55, 0.03, 0.90, 0.08, 330, 1.35, 0.90),
]


def validate_config(seed, years, stores, out):
    """проверяет параметры запуска. Бросает InvalidConfigError"""
    if seed < 0:
        raise InvalidConfigError("--seed должен быть неотрицательным")
    if not 1 <= years <= MAX_YEARS:
        raise InvalidConfigError(f"--years должно быть от 1 до {MAX_YEARS}")
    if not MIN_STORES <= stores <= MAX_STORES:
        raise InvalidConfigError(
            f"--stores должно быть от {MIN_STORES} до {MAX_STORES}")
    if not out or not out.strip():
        raise InvalidConfigError("--out: имя файла не может быть пустым")
    if os.path.isdir(out):
        raise InvalidConfigError(f"--out: '{out}' является папкой, нужен файл")
    if not out.lower().endswith(".csv"):
        raise InvalidConfigError("--out: файл должен иметь расширение .csv")


def make_promo_flags(rng, shape, start_prob=0.02, length=7):
    """генерирует акции длительностью `length` дней"""
    starts = rng.random(shape) < start_prob
    csum = np.cumsum(starts, axis=0)
    lagged = np.zeros_like(csum)
    lagged[length:] = csum[:-length]
    promo = (csum - lagged) > 0          # был старт за последние `length` дней

    after = np.zeros_like(promo)
    for k in range(1, 4):
        shifted = np.zeros_like(promo)
        shifted[k:] = promo[:-k]
        after |= shifted
    after &= ~promo
    return promo, after


def simulate_sales(rng, years, n_stores):
    """моделирует продажи: тренд, сезонности, акции, шум, способы оплаты"""
    start = pd.Timestamp(START_DATE)
    dates = pd.date_range(start, start + pd.DateOffset(years=years)
                          - pd.Timedelta(days=1), freq="D")
    n_days, n_prod = len(dates), len(PRODUCTS)
    shape = (n_days, n_stores, n_prod)

    t = ((dates - dates[0]).days.to_numpy() / 365.25)[:, None, None]
    weekend = (dates.dayofweek.to_numpy() >= 5)[:, None, None]
    doy = dates.dayofyear.to_numpy()[:, None, None]

    def col(attr):
        return np.array([getattr(p, attr) for p in PRODUCTS],
                        dtype=float)[None, None, :]

    store_size = rng.uniform(0.7, 1.4, n_stores)[None, :, None]
    city_price = np.array([CITIES[i][1] for i in range(n_stores)])[None, :, None]

    trend = 1 + col("trend") * t
    week = np.where(weekend, col("weekend"), 1.0)
    season = 1 + col("season_amp") * np.cos(
        2 * np.pi * (doy - col("peak_day")) / 365.25)

    promo, after = make_promo_flags(rng, shape)

    lam = col("base_qty") * store_size * trend * week * season
    lam = lam * np.where(promo, col("promo_lift"), 1.0) \
              * np.where(after, 0.85, 1.0)
    qty = np.maximum(rng.poisson(lam), 1)

    price = (col("base_price") * (1 + 0.05 * t) * city_price
             * rng.normal(1, 0.02, shape)
             * np.where(promo, col("promo_discount"), 1.0))
    cost = (col("base_price") * col("cost_ratio") * (1 + 0.04 * t)
            * rng.normal(1, 0.01, shape))

    frac = np.broadcast_to(t / t.max(), shape)      # доля пройденного времени
    u = rng.random(shape)
    cash_p = 0.40 - 0.15 * frac                     # наличные падают
    online_p = 0.10 + 0.20 * frac                   # онлайн растёт
    payment = np.where(u < cash_p, "наличные",
                       np.where(u < 1 - online_p, "карта", "онлайн"))

    store_ids = np.array([f"S{i + 1:02d}" for i in range(n_stores)])
    cities = np.array([CITIES[i][0] for i in range(n_stores)])
    names = np.array([p.name for p in PRODUCTS])
    cats = np.array([p.category for p in PRODUCTS])

    return pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), n_stores * n_prod),
        "store_id": np.tile(np.repeat(store_ids, n_prod), n_days),
        "city": np.tile(np.repeat(cities, n_prod), n_days),
        "category": np.tile(cats, n_days * n_stores),
        "product": np.tile(names, n_days * n_stores),
        "qty": qty.ravel(),
        "price": price.ravel().round(2),
        "cost": cost.ravel().round(2),
        "promo": promo.ravel().astype(int),
        "payment_method": payment.ravel(),
    })


def inject_anomalies(df, rng, rate=OUTLIER_RATE):
    """добавляет выбросы: всплески qty и аномальные цены"""
    n = len(df)
    idx = rng.choice(n, int(n * rate), replace=False)
    q_idx, p_idx = idx[: len(idx) // 2], idx[len(idx) // 2:]

    qty = df["qty"].to_numpy().copy()
    qty[q_idx] = qty[q_idx] * rng.integers(4, 10, size=len(q_idx))
    df["qty"] = qty

    price = df["price"].to_numpy().copy()
    price[p_idx] = (price[p_idx] * rng.uniform(2.5, 4.0, size=len(p_idx))).round(2)
    df["price"] = price
    return df


def inject_missing(df, rng, rate=MISSING_RATE):
    """случайно удаляет значения в price и payment_method."""
    n = len(df)
    for column in ("price", "payment_method"):
        idx = rng.choice(n, int(n * rate), replace=False)
        df.iloc[idx, df.columns.get_loc(column)] = np.nan
    return df


def generate_dataset(seed, years, stores):
    """полный набор данных для заданных параметров """
    rng = np.random.default_rng(seed)
    df = simulate_sales(rng, years, stores)
    df = inject_anomalies(df, rng)
    df = inject_missing(df, rng)
    return df


def save_csv(df, path):
    """сохраняет DataFrame в CSV. Бросает OutputError при проблемах с записью  """
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
    except OSError as err:
        raise OutputError(
            f"не удалось записать файл '{path}': {err.strerror or err}") from err


def parse_args(argv=None):
    """разбирает аргументы командной строки"""
    parser = argparse.ArgumentParser(
        description="Генератор синтетических данных о продажах сети магазинов")
    parser.add_argument("--seed", type=int, default=42,
                        help="зерно генератора (по умолчанию 42)")
    parser.add_argument("--years", type=int, default=3,
                        help=f"число лет, 1..{MAX_YEARS} (по умолчанию 3)")
    parser.add_argument("--stores", type=int, default=6,
                        help=f"число магазинов, {MIN_STORES}..{MAX_STORES} "
                             "(по умолчанию 6)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="путь к выходному CSV (по умолчанию data/sales.csv)")
    return parser.parse_args(argv)


def main(argv=None):
    """точка входа. Возвращает код возврата (0 - успех)"""
    args = parse_args(argv)
    try:
        validate_config(args.seed, args.years, args.stores, args.out)
        df = generate_dataset(args.seed, args.years, args.stores)
        save_csv(df, args.out)
    except GeneratorError as err:
        print(f"Ошибка: {err}", file=sys.stderr)
        return 1
    except Exception as err:  # непредвиденные ситуации тоже без traceback
        print(f"Непредвиденная ошибка: {err}", file=sys.stderr)
        return 2

    print(f"Готово: {args.out}")
    print(f"Строк: {len(df)}, период: {df['date'].min():%Y-%m-%d} - "
          f"{df['date'].max():%Y-%m-%d}")
    print(f"Пропусков price: {df['price'].isna().mean():.2%}, "
          f"payment_method: {df['payment_method'].isna().mean():.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())