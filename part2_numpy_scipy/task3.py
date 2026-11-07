import numpy as np

A = np.array([[4.0, -1.5, 2.0, 0.5],
              [1.0, 5.0, -2.0, 1.2],
              [-2.0, 1.0, 6.0, -1.0],
              [0.5, 2.0, -1.0, 7.0]])
B = np.array([[11.05], [-6.82], [11.5], [-0.5]])


def main():
    det = np.linalg.det(A)
    print(f"det(A) = {det:.4f}")
    if np.isclose(det, 0):
        print("Определитель равен нулю, система не имеет единственного решения.")
        return
    print(f"cond(A) = {np.linalg.cond(A):.4f}")

    x_inv = np.linalg.inv(A) @ B
    x_solve = np.linalg.solve(A, B)

    print("\nX = A^-1 * B:   ", np.round(x_inv.ravel(), 1))
    print("np.linalg.solve:", np.round(x_solve.ravel(), 1))

    print(f"\n||X_inv - X_solve|| = {np.linalg.norm(x_inv - x_solve):.3e}")
    print(f"||A*X_inv - B||     = {np.linalg.norm(A @ x_inv - B):.3e}")
    print(f"||A*X_solve - B||   = {np.linalg.norm(A @ x_solve - B):.3e}")

    print("\nОтвет:", np.round(x_solve.ravel(), 1))


if __name__ == "__main__":
    main()