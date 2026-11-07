import os
import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")


def p(x_deg):
    """p(x) = e^(sin x) + ln(sin^2(0.5x) + 1) * cos x, x в градусах"""
    x = np.deg2rad(x_deg)
    return np.exp(np.sin(x)) + np.log(np.sin(0.5 * x) ** 2 + 1) * np.cos(x)


def q(x_deg):
    """q(x) = -ln((sin x - cos x)^2 + 2) + 8, x в градусах."""
    x = np.deg2rad(x_deg)
    return -np.log((np.sin(x) - np.cos(x)) ** 2 + 2) + 8


def main():
    x = np.linspace(-360, 360, 2000)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    ax1.plot(x, p(x), color="tab:blue", label="p(x)")
    ax1.set_title("График функции p(x)")
    ax1.set_ylabel("p(x)")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(x, q(x), color="tab:red", label="q(x)")
    ax2.set_title("График функции q(x)")
    ax2.set_xlabel("x, градусы")
    ax2.set_ylabel("q(x)")
    ax2.grid(True)
    ax2.legend()

    ax2.set_xticks(np.arange(-360, 361, 90))
    ax2.set_xlim(-360, 360)

    fig.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, "task1.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()