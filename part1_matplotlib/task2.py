import os
import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")


def f(x):
    """Вычисляет 4/(x^2-4) окрестность асимптот x=±2 маскируется"""
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 4 / (x ** 2 - 4)
    near_asymptote = np.abs(np.abs(x) - 2) < 0.03
    return np.ma.masked_where(near_asymptote | ~np.isfinite(y), y)


def main():
    x = np.linspace(-8, 8, 4001)
    y = f(x)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, color="tab:green", label=r"$y=\frac{4}{x^2-4}$")
    for a in (-2, 2):
        ax.axvline(a, linestyle="--", color="gray",
                   label=f"асимптота x = {a}" if a == -2 else None)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_ylim(-10, 10)
    ax.set_xlim(-8, 8)
    ax.set_title("Функция с разрывами")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()

    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, "task2.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()