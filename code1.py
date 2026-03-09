import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# REAL-WORLD MODEL:
# Bank account with adaptive interest rate
#
# B(t) = bank balance
# r(t) = interest rate
#
# SYSTEM OF ODEs:
# dB/dt = r * B
# dr/dt = k * (Bd - B)
#
# This is a system of ordinary differential equations (ODEs)
# ============================================================


# -----------------------
# MODEL PARAMETERS
# -----------------------
Bd = 2000.0     # Desired (target) balance B_d
k = 0.001       # Reaction speed (how fast interest rate changes)

# -----------------------
# TIME DISCRETIZATION
# -----------------------
h = 0.1         # Time step size
T = 50          # Final time
N = int(T / h)  # Number of time steps


# -----------------------
# INITIAL CONDITIONS (IVP)
# -----------------------
# B(0) = initial balance
# r(0) = initial interest rate
B = np.zeros(N + 1)
r = np.zeros(N + 1)

B[0] = 1000.0
r[0] = 0.01


# ============================================================
# IMPLICIT EULER METHOD
#
# General implicit Euler formula:
# y_{n+1} = y_n + h * f(y_{n+1})
#
# Applied to our system:
#
# B_{n+1} = B_n + h * r_{n+1} * B_{n+1}
# r_{n+1} = r_n + h * k * (Bd - B_{n+1})
#
# This is a NONLINEAR algebraic system
# ============================================================


# -----------------------
# FIXED-POINT ITERATION PARAMETERS
# -----------------------
max_iter = 50
tol = 1e-6


# -----------------------
# TIME STEPPING LOOP
# -----------------------
for n in range(N):

    # Initial guess for Fixed-Point iteration
    B_new = B[n]
    r_new = r[n]

    # -----------------------
    # FIXED-POINT ITERATION
    # -----------------------
    for _ in range(max_iter):

        B_old = B_new
        r_old = r_new

        # Fixed-point update formulas:
        #
        # B_{n+1}^{(k+1)} = B_n + h * r_{n+1}^{(k)} * B_{n+1}^{(k)}
        # r_{n+1}^{(k+1)} = r_n + h * k * (Bd - B_{n+1}^{(k+1)})
        #
        B_new = B[n] + h * r_old * B_old
        r_new = r[n] + h * k * (Bd - B_new)

        # Convergence check
        if abs(B_new - B_old) < tol and abs(r_new - r_old) < tol:
            break

    # Save converged values
    B[n + 1] = B_new
    r[n + 1] = r_new


# -----------------------
# VISUALIZATION
# -----------------------
t = np.linspace(0, T, N + 1)

plt.plot(t, B, label="Balance B(t)")
plt.plot(t, r, label="Interest rate r(t)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Implicit Euler + Fixed-Point Iteration")
plt.legend()
plt.grid()
plt.show()
