import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Interactive user input with default values
# ============================================================

def get_input(prompt, default):
    val = input(f"{prompt} [default {default}]: ")
    return float(val) if val.strip() != "" else default

print("\n=== Car Acceleration Simulator ===")
m   = get_input("Car mass (kg)", 1400.0)
Cd  = get_input("Drag coefficient Cd", 0.32)
A   = get_input("Frontal area A (m^2)", 2.2)
rho = get_input("Air density (kg/m^3)", 1.225)
Crr = get_input("Rolling resistance Crr", 0.015)
g   = 9.81  # constant

P_max = get_input("Max engine power (W)", 80000.0)
F_max = get_input("Max traction force (N)", 4000.0)

T = get_input("Simulation time (s)", 20.0)

# dt list input
dt_user = input("Enter dt list (comma separated) [default 0.5,0.2,0.1,0.05,0.02]: ")
if dt_user.strip() == "":
    dts = [0.5, 0.2, 0.1, 0.05, 0.02]
else:
    dts = [float(x) for x in dt_user.split(",")]

v0 = get_input("Initial velocity (m/s)", 0.0)
x0 = get_input("Initial position (m)", 0.0)

dt_ref = get_input("Reference dt (for RK4)", 1e-4)

print("\nRunning simulation...\n")

# ============================================================
# Simulation Code
# ============================================================

def forces(v):
    F_drag = 0.5 * rho * Cd * A * v**2
    F_roll = m * g * Crr
    F_trac = min(F_max, P_max / max(v, 1e-3))
    return F_trac, F_drag, F_roll

def dvdt(v):
    F_trac, F_drag, F_roll = forces(v)
    return (F_trac - F_drag - F_roll) / m

def f_state(y):
    v = y[0]
    return np.array([dvdt(v), v])

def euler_step(y, h):
    return y + h * f_state(y)

def heun_step(y, h):
    k1 = f_state(y)
    k2 = f_state(y + h*k1)
    return y + 0.5*h*(k1 + k2)

def rk4_step(y, h):
    k1 = f_state(y)
    k2 = f_state(y + 0.5*h*k1)
    k3 = f_state(y + 0.5*h*k2)
    k4 = f_state(y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate(step_func, y0, t):
    y = np.array(y0, dtype=float)
    sol = np.zeros((len(t), 2))
    sol[0] = y
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        y = step_func(y, h)
        sol[i] = y
    return sol

# Reference solution
y0 = [v0, x0]
t_ref = np.arange(0, T+dt_ref, dt_ref)
sol_ref = integrate(rk4_step, y0, t_ref)
v_ref = sol_ref[:,0]

methods = [('Euler', euler_step), ('Heun', heun_step), ('RK4', rk4_step)]
errors = {name: [] for name, _ in methods}

# MAIN loop
for dt in dts:
    t = np.arange(0, T+dt, dt)
    for name, func in methods:
        sol = integrate(func, y0, t)
        v_sol = sol[:,0]
        v_ref_at_t = np.interp(t, t_ref, v_ref)
        err = np.sqrt(np.mean((v_sol - v_ref_at_t)**2))
        errors[name].append(err)

    if abs(dt - dts[-1]) < 1e-12:
        plt.figure(figsize=(10,5))
        for name, func in methods:
            sol = integrate(func, y0, t)
            plt.plot(t, sol[:,0], label=name)
        plt.plot(t_ref[::200], v_ref[::200], '--', label='RK4 ref')
        plt.xlabel('t (s)')
        plt.ylabel('v (m/s)')
        plt.legend()
        plt.grid(True)
        plt.title(f'Car Speed Comparison (dt={dt})')
        plt.show()

# Error plot
plt.figure()
for name in errors:
    plt.loglog(dts, errors[name], 'o-', label=name)
plt.xlabel('dt (s)')
plt.ylabel('L2 error on v')
plt.legend()
plt.grid(True, which='both')
plt.title('Error vs dt (log-log)')
plt.show()

# Print numerical errors
print("dt\tEuler\t\tHeun\t\tRK4")
for i, dt in enumerate(dts):
    print(f"{dt:.3f}\t{errors['Euler'][i]:.3e}\t{errors['Heun'][i]:.3e}\t{errors['RK4'][i]:.3e}")

# Time to 100 km/h
v_target = 27.78
idx = np.argmax(v_ref >= v_target)
if v_ref[idx] >= v_target:
    print(f"\nReference time to reach 100 km/h ≈ {t_ref[idx]:.2f} s\n")
else:
    print("\nCar did not reach 100 km/h.\n")


