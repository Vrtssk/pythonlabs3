import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse, Arc

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")


def main():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Rectangle((0, 0), 20, 12, color="#dbeafe", zorder=0))
    ax.add_patch(Rectangle((0, 0), 20, 3, color="#86c98a", zorder=1))

    ax.add_patch(Circle((16.5, 10), 1.2, color="#fcd34d", zorder=1))

    ax.add_patch(Ellipse((5, 10), 4, 1.2, color="white", zorder=1))
    ax.add_patch(Ellipse((6.2, 10.5), 2.4, 1.1, color="white", zorder=1))

    ax.add_patch(Rectangle((2, 3), 4.5, 6.5, color="#94a3b8", zorder=2))
    for y in (4.2, 6.2, 8.0):
        ax.add_patch(Rectangle((3.2, y), 2.1, 0.9, color="white", zorder=3))

    ax.add_patch(Rectangle((10, 3), 5, 3, color="#fda4af", zorder=2))
    ax.add_patch(Polygon([(9.5, 6), (15.5, 6), (12.5, 8.3)],
                         color="#be123c", zorder=2))
    ax.add_patch(Rectangle((12, 3), 1, 1.8, color="white", zorder=3))

    ax.add_patch(Arc((8, 3), 2, 1.6, theta1=0, theta2=180,
                     color="#2f855a", linewidth=4, zorder=2))

    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(os.path.join(IMG_DIR, "task3.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()