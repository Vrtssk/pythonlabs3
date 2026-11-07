import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")


def single_integral():
    """интеграл от x*sin(x) на [0, pi]"""
    value, _ = integrate.quad(lambda x: x * np.sin(x), 0, np.pi)
    return value


def double_integral():
    """интеграл (x+y) по области 0 <= y <= x <= 1"""
    value, _ = integrate.dblquad(lambda y, x: x + y, 0, 1,
                                 lambda x: 0, lambda x: x)
    return value


def monte_carlo(n, rng):
    """оценка двойного интеграла методом Монте-Карло по n точкам"""
    x = rng.random(n)
    y = rng.random(n)
    inside = y <= x
    return np.mean((x + y) * inside)  # площадь квадрата [0,1]^2 равна 1


def main():
    i1 = single_integral()
    i2 = double_integral()
    print(f"quad:    {i1:.10f}, аналитически {np.pi:.10f}, "
          f"погрешность {abs(i1 - np.pi):.2e}")
    print(f"dblquad: {i2:.10f}, аналитически 0.5, "
          f"погрешность {abs(i2 - 0.5):.2e}")

    rng = np.random.default_rng(42)
    ns = np.array([10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6])
    estimates = np.array([monte_carlo(int(n), rng) for n in ns])
    errors = np.abs(estimates - 0.5)

    print("\nМонте-Карло:")
    print(f"{'N':>10} {'оценка':>12} {'погрешность':>14}")
    print("\n".join(f"{n:>10} {e:>12.6f} {err:>14.2e}"
                    for n, e, err in zip(ns, estimates, errors)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ns, errors, "o-", label="погрешность Монте-Карло")
    ax.loglog(ns, 1 / np.sqrt(ns), "--", color="gray", label=r"$1/\sqrt{N}$")
    ax.set_title("Погрешность метода Монте-Карло от N")
    ax.set_xlabel("N (число точек)")
    ax.set_ylabel("|оценка - 0.5|")
    ax.grid(True, which="both")
    ax.legend()

    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, "task4_mc_error.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()