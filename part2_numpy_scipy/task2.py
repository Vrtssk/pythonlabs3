import numpy as np


def parse_numbers(line):
    """разбирает строку в массив чисел. При ошибке бросает ValueError."""
    parts = line.replace(",", ".").split()
    if not parts:
        raise ValueError("строка пуста, введите числа через пробел")
    try:
        arr = np.array(parts, dtype=float)
    except ValueError:
        raise ValueError("допустимы только числа, разделённые пробелами")
    if not np.all(np.isfinite(arr)):
        raise ValueError("значения должны быть конечными числами")
    return arr


def read_array(prompt, name):
    while True:
        try:
            arr = parse_numbers(input(prompt))
            if np.any(arr <= 0):
                raise ValueError(f"{name} должны быть больше нуля")
            return arr
        except ValueError as err:
            print(f"Ошибка: {err}. Повторите ввод.")


def read_int(prompt, low, high, name):
    """запрашивает целое число в диапазоне [low, high]"""
    while True:
        try:
            value = float(input(prompt).replace(",", "."))
        except ValueError:
            print(f"Ошибка: {name} должно быть целым числом. Повторите ввод.")
            continue
        if not value.is_integer():
            print(f"Ошибка: {name} должно быть целым числом. Повторите ввод.")
        elif not low <= value <= high:
            print(f"Ошибка: {name} должно быть от {low} до {high}. Повторите ввод.")
        else:
            return int(value)


def route_stats(lengths, speeds, k, p):
    """считает путь, время и среднюю скорость на участках k..p"""
    seg_len = lengths[k - 1:p]
    seg_speed = speeds[k - 1:p]
    seg_time = seg_len / seg_speed
    s = seg_len.sum()
    t = seg_time.sum()
    return s, t, s / t, seg_len, seg_speed, seg_time


def position_after(seg_time, seg_speed, k, t):
    """находит номер участка и расстояние от его начала через t часов"""
    ends = np.cumsum(seg_time)
    starts = ends - seg_time
    idx = int(np.searchsorted(ends, t, side="left"))
    idx = min(idx, len(ends) - 1)
    dist = (t - starts[idx]) * seg_speed[idx]
    return k + idx, dist


def main():
    while True:
        lengths = read_array("Длины участков (через пробел): ", "длины")
        speeds = read_array("Скорости на участках (через пробел): ", "скорости")
        if len(lengths) == len(speeds):
            break
        print(f"Ошибка: длин {len(lengths)}, а скоростей {len(speeds)}. "
              "Количество должно совпадать. Повторите ввод.")

    n = len(lengths)
    while True:
        k = read_int(f"k (участок въезда, 1..{n}): ", 1, n, "k")
        p = read_int(f"p (участок выезда, {k}..{n}): ", 1, n, "p")
        if k <= p:
            break
        print("Ошибка: k не может быть больше p. Повторите ввод k и p.")

    s, total_time, v, _, seg_speed, seg_time = route_stats(lengths, speeds, k, p)

    while True:
        try:
            t = float(input(f"t (часов после въезда, 0..{total_time:.2f}): ")
                      .replace(",", "."))
        except ValueError:
            print("Ошибка: t должно быть числом. Повторите ввод.")
            continue
        if not np.isfinite(t) or t < 0:
            print("Ошибка: t должно быть неотрицательным числом. Повторите ввод.")
        elif t > total_time:
            print(f"Ошибка: t больше времени в пути ({total_time:.2f} ч). "
                  "Повторите ввод.")
        else:
            break

    seg_no, dist = position_after(seg_time, seg_speed, k, t)
    print(f"S = {s:g} км, T = {total_time:.2f} ч, V = {v:.2f} км/ч")
    print(f"Через {t:g} ч автомобиль на участке {seg_no}, "
          f"в {dist:.1f} км от его начала")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nВвод прерван.")