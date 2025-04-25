import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

zDes = 100.0
MASS = 2
g = 9.81

uMax = 4 * g
uMin = 0

vl = []
vlDot = []

def altitudeControl(t, state):
    z, v, eI, kP, kI, kD = state

    # Error terms
    eZ = zDes - z
    eV = -v

    # Control
    u = kP * eZ + kI * eI + kD * eV
    u = max(min(u, uMax), uMin)

    print(u)

    # Lyapunov function
    vl = 0.5 * eZ**2 + 0.5 * eV**2 + 0.5 * eI**2
    vlDot = eZ * (-v) + eV * (-u / MASS + g) + eI * eZ

    # Adaptive gain update
    damping_factor = 0.000001
    kPDot = damping_factor * vlDot * kP
    kIDot = damping_factor * vlDot * kI
    kDDot = damping_factor * vlDot * kD

    # Dynamics
    zDot = v
    vDot = u / MASS - g

    eIDot = eZ

    return [zDot, vDot, eIDot, 0, 0, 0]

# Initial state: [z, v, eI, kP, kI, kD]
initState = [10.0, 0, 0, 2.0, 0.1, 6.0]  # Starting with base PID gains

# Time span for integration
t_span = (0, 100)  # Time from 0 to 100 seconds

# Solve the system
sol = solve_ivp(altitudeControl, t_span, initState)

# Do Lyapunov Stabilty Analysis
z, v, eI, kP, kI, kD = sol.y
eZ = zDes - z
eV = -v
u = kP * eZ + kI * eI + kD * eV
u = np.clip(u, uMin, uMax)
vl = 0.5 * eZ**2 + 0.5 * eV**2 + 0.5 * eI**2
vlDot = eZ * (-v) + eV * (-u / MASS + g) + eI * eZ

# Plot altitude, velocity, and Lyapunov function over time
plt.figure(figsize=(10, 8))

# Altitude plot
plt.subplot(2, 1, 1)
plt.plot(sol.t, sol.y[0, :], label='Altitude (z)')
plt.title('Altitude Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.grid(True)

# Velocity plot
plt.subplot(2, 1, 2)
plt.plot(sol.t, sol.y[1, :], label='Velocity (v)')
plt.title('Velocity Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)

plt.tight_layout()

# Now, plot Lyapunov stability analysis separately
plt.figure(figsize=(10, 8))

# Lyapunov function plot
plt.subplot(2, 1, 1)
plt.plot(sol.t, vl, label='Lyapunov Function (V)')
plt.title('Lyapunov Function Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Lyapunov Function Value')
plt.grid(True)

# Lyapunov derivative plot
plt.subplot(2, 1, 2)
plt.plot(sol.t, vlDot, label='Lyapunov Derivative (V dot)', color='red')
plt.title('Lyapunov Derivative Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Lyapunov Derivative Value')
plt.grid(True)

plt.tight_layout()
plt.show()
