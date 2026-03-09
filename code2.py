import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PROJECT: Numerical solution of a system of ODEs
# METHOD: Implicit Euler + Newton–Gauss–Seidel
#
# REAL-WORLD MODEL:
#   Bank account with adaptive interest rate
#
# UNKNOWN FUNCTIONS:
#   B(t) - bank balance
#   r(t) - interest rate
#
# SYSTEM OF ORDINARY DIFFERENTIAL EQUATIONS (ODEs):
#
#   (1) dB/dt = r * B
#   (2) dr/dt = k * (Bd - B)
#
# where:
#   Bd = desired (target) balance
#   k  = reaction parameter
#
# This is a NONLINEAR system because of the product r * B
# ============================================================


# ------------------------------------------------------------
# MODEL PARAMETERS
# ------------------------------------------------------------
Bd = 2000.0   # Desired balance B_d
k = 0.001     # Speed of interest-rate adjustment


# ------------------------------------------------------------
# TIME DISCRETIZATION
# ------------------------------------------------------------
# We solve the problem on the interval [0, T]
# using uniform time steps of size h
h = 0.1
T = 50
N = int(T / h)   # Number of time steps


# ------------------------------------------------------------
# INITIAL VALUE PROBLEM (IVP)
# ------------------------------------------------------------
# Initial conditions:
#   B(0) = B0
#   r(0) = r0
B = np.zeros(N + 1)
r = np.zeros(N + 1)

B[0] = 1000.0   # Initial balance
r[0] = 0.01     # Initial interest rate


# ============================================================
# IMPLICIT EULER METHOD
#
# General implicit Euler formula for ODE y' = f(y):
#
#   y_{n+1} = y_n + h * f(y_{n+1})
#
# Applying implicit Euler to our system gives:
#
#   B_{n+1} = B_n + h * r_{n+1} * B_{n+1}
#   r_{n+1} = r_n + h * k * (Bd - B_{n+1})
#
# This is a NONLINEAR ALGEBRAIC SYSTEM
# for the unknowns (B_{n+1}, r_{n+1})
# ============================================================


# ============================================================
# NEWTON–GAUSS–SEIDEL METHOD
#
# To apply Newton's method, we rewrite the system as:
#
#   F1(B, r) = B - B_n - h * r * B = 0
#   F2(B, r) = r - r_n - h * k * (Bd - B) = 0
#
# At each Newton iteration, we solve:
#
#   J(B, r) * [dB, dr]^T = -[F1, F2]^T
#
# where J is the Jacobian matrix of partial derivatives
# ============================================================


# ------------------------------------------------------------
# NEWTON ITERATION PARAMETERS
# ------------------------------------------------------------
max_iter = 20      # Maximum Newton iterations per time step
tol = 1e-6         # Convergence tolerance


# ------------------------------------------------------------
# MAIN TIME-STEPPING LOOP
# ------------------------------------------------------------
for n in range(N):

    # Initial guess for Newton iteration:
    # using previous time-step values
    B_new = B[n]
    r_new = r[n]

    # --------------------------------------------------------
    # NEWTON–GAUSS–SEIDEL ITERATION
    # --------------------------------------------------------
    for _ in range(max_iter):

        # ----------------------------------------------------
        # NONLINEAR FUNCTIONS F1, F2
        #
        # F1(B, r) = B - B_n - h * r * B
        # F2(B, r) = r - r_n - h * k * (Bd - B)
        # ----------------------------------------------------
        F1 = B_new - B[n] - h * r_new * B_new
        F2 = r_new - r[n] - h * k * (Bd - B_new)

        # ----------------------------------------------------
        # JACOBIAN MATRIX J
        #
        # J = [ dF1/dB   dF1/dr ]
        #     [ dF2/dB   dF2/dr ]
        #
        # Partial derivatives:
        #
        # dF1/dB = 1 - h * r
        # dF1/dr = -h * B
        # dF2/dB = h * k
        # dF2/dr = 1
        # ----------------------------------------------------
        J11 = 1 - h * r_new
        J12 = -h * B_new
        J21 = h * k
        J22 = 1

        # ----------------------------------------------------
        # SOLVE THE LINEAR SYSTEM:
        #
        #   [J11 J12] [dB] = [-F1]
        #   [J21 J22] [dr]   [-F2]
        #
        # For a 2x2 system, we solve it explicitly
        # ----------------------------------------------------
        det = J11 * J22 - J12 * J21

        dB = (-F1 * J22 + F2 * J12) / det
        dr = (-F2 * J11 + F1 * J21) / det

        # ----------------------------------------------------
        # GAUSS–SEIDEL UPDATE
        #
        # New solution = old solution + Newton correction
        # ----------------------------------------------------
        B_new += dB
        r_new += dr

        # ----------------------------------------------------
        # CONVERGENCE CHECK
        #
        # Stop if Newton corrections are sufficiently small
        # ----------------------------------------------------
        if abs(dB) < tol and abs(dr) < tol:
            break

    # Save converged solution for time step n+1
    B[n + 1] = B_new
    r[n + 1] = r_new


# ------------------------------------------------------------
# VISUALIZATION OF RESULTS
# ------------------------------------------------------------
t = np.linspace(0, T, N + 1)

plt.plot(t, B, label="Balance B(t)")
plt.plot(t, r, label="Interest rate r(t)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Implicit Euler + Newton–Gauss–Seidel")
plt.legend()
plt.grid()
plt.show()
