import numpy as np

EXPENSES = np.array([3200, 3100, 2600, 1900, 1500, 1300,
                     1250, 1400, 1800, 2300, 2900, 3500])

WINTER_IDX = np.array([11, 0, 1])
SUMMER_IDX = np.array([5, 6, 7])


def season_sums(expenses):
    """возвращает суммарные расходы за зиму и за лето"""
    return expenses[WINTER_IDX].sum(), expenses[SUMMER_IDX].sum()


def max_months(expenses):
    """возвращает номера месяцев (с 1) в которые расходы максимальны"""
    return np.flatnonzero(expenses == expenses.max()) + 1


def main():
    print("Расходы:", *EXPENSES)
    winter, summer = season_sums(EXPENSES)
    if winter > summer:
        verdict = "больше тратится зимой"
    elif winter < summer:
        verdict = "больше тратится летом"
    else:
        verdict = "расходы зимой и летом равны"
    print(f"Зима: {winter}, лето: {summer} -> {verdict}")

    months = max_months(EXPENSES)
    print(f"Максимум ({EXPENSES.max()}) в месяцах:", *months)


if __name__ == "__main__":
    main()