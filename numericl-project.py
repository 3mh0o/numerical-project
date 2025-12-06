import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================
# Simulation Functions
# ==========================
def forces(v, m, Cd, A, rho, Crr, g, P_max, F_max):
    F_drag = 0.5 * rho * Cd * A * v**2
    F_roll = m * g * Crr
    F_trac = min(F_max, P_max / max(v, 1e-3))
    return F_trac, F_drag, F_roll

def dvdt(v, m, Cd, A, rho, Crr, g, P_max, F_max):
    F_trac, F_drag, F_roll = forces(v, m, Cd, A, rho, Crr, g, P_max, F_max)
    return (F_trac - F_drag - F_roll) / m

def f_state(y, m, Cd, A, rho, Crr, g, P_max, F_max):
    v = y[0]
    return np.array([dvdt(v, m, Cd, A, rho, Crr, g, P_max, F_max), v])

def euler_step(y, h, *args):
    return y + h * f_state(y, *args)

def heun_step(y, h, *args):
    k1 = f_state(y, *args)
    k2 = f_state(y + h*k1, *args)
    return y + 0.5*h*(k1 + k2)

def rk4_step(y, h, *args):
    k1 = f_state(y, *args)
    k2 = f_state(y + 0.5*h*k1, *args)
    k3 = f_state(y + 0.5*h*k2, *args)
    k4 = f_state(y + h*k3, *args)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate(step_func, y0, t, *args):
    y = np.array(y0, dtype=float)
    sol = np.zeros((len(t), 2))
    sol[0] = y
    for i in range(1, len(t)):
        h = t[i] - t[i-1]
        y = step_func(y, h, *args)
        sol[i] = y
    return sol

# ==========================
# GUI
# ==========================
class CarSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Acceleration Simulator")

        self.entries = {}
        params = [
            ("Car mass (kg)", 1400.0),
            ("Drag coefficient Cd", 0.32),
            ("Frontal area A (m^2)", 2.2),
            ("Air density rho (kg/m^3)", 1.225),
            ("Rolling resistance Crr", 0.015),
            ("Max engine power P_max (W)", 80000.0),
            ("Max traction force F_max (N)", 4000.0),
            ("Simulation time T (s)", 20.0),
            ("Initial velocity v0 (m/s)", 0.0),
            ("Initial position x0 (m)", 0.0),
            ("Reference dt (for RK4)", 1e-4),
            ("dt list (comma separated)", "0.5,0.2,0.1,0.05,0.02")
        ]

        for i, (label, default) in enumerate(params):
            tk.Label(root, text=label).grid(row=i, column=0, sticky="w")
            e = tk.Entry(root)
            e.grid(row=i, column=1)
            e.insert(0, str(default))
            self.entries[label] = e

        tk.Button(root, text="Run Simulation", command=self.run_sim).grid(row=len(params), column=0, columnspan=2, pady=10)

        # Text box for results
        self.text = tk.Text(root, height=15, width=60)
        self.text.grid(row=len(params)+1, column=0, columnspan=2)

        # Figure for speed curves
        self.fig1, self.ax1 = plt.subplots(figsize=(6,4))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=root)
        self.canvas1.get_tk_widget().grid(row=0, column=2, rowspan=len(params)+1)

        # Figure for error log-log plot
        self.fig2, self.ax2 = plt.subplots(figsize=(6,4))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=root)
        self.canvas2.get_tk_widget().grid(row=len(params)+1, column=2, rowspan=8)

    def run_sim(self):
        # Read input values
        m = float(self.entries["Car mass (kg)"].get())
        Cd = float(self.entries["Drag coefficient Cd"].get())
        A = float(self.entries["Frontal area A (m^2)"].get())
        rho = float(self.entries["Air density rho (kg/m^3)"].get())
        Crr = float(self.entries["Rolling resistance Crr"].get())
        P_max = float(self.entries["Max engine power P_max (W)"].get())
        F_max = float(self.entries["Max traction force F_max (N)"].get())
        T = float(self.entries["Simulation time T (s)"].get())
        v0 = float(self.entries["Initial velocity v0 (m/s)"].get())
        x0 = float(self.entries["Initial position x0 (m)"].get())
        dt_list = [float(x) for x in self.entries["dt list (comma separated)"].get().split(",")]
        dt_ref = float(self.entries["Reference dt (for RK4)"].get())
        g = 9.81

        y0 = [v0, x0]
        t_ref = np.arange(0, T+dt_ref, dt_ref)
        sol_ref = integrate(rk4_step, y0, t_ref, m, Cd, A, rho, Crr, g, P_max, F_max)
        v_ref = sol_ref[:,0]

        methods = [('Euler', euler_step), ('Heun', heun_step), ('RK4', rk4_step)]
        errors = {name: [] for name, _ in methods}

        # Calculate errors for different dt
        for dt in dt_list:
            t = np.arange(0, T+dt, dt)
            for name, func in methods:
                sol = integrate(func, y0, t, m, Cd, A, rho, Crr, g, P_max, F_max)
                v_sol = sol[:,0]
                v_ref_at_t = np.interp(t, t_ref, v_ref)
                err = np.sqrt(np.mean((v_sol - v_ref_at_t)**2))
                errors[name].append(err)

        # ---------------------
        # Plot speed curves
        # ---------------------
        t = np.arange(0, T+dt_list[-1], dt_list[-1])
        self.ax1.clear()
        for name, func in methods:
            sol = integrate(func, y0, t, m, Cd, A, rho, Crr, g, P_max, F_max)
            self.ax1.plot(t, sol[:,0], label=name)
        self.ax1.plot(t_ref[::200], v_ref[::200], '--', label='RK4 ref')
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Velocity (m/s)")
        self.ax1.set_title(f"Car Speed Comparison (dt={dt_list[-1]})")
        self.ax1.grid(True)
        self.ax1.legend()
        self.canvas1.draw()

        # ---------------------
        # Plot error log-log
        # ---------------------
        self.ax2.clear()
        for name in errors:
            self.ax2.loglog(dt_list, errors[name], 'o-', label=name)
        self.ax2.set_xlabel("dt (s)")
        self.ax2.set_ylabel("L2 error on velocity")
        self.ax2.set_title("Error vs dt (log-log)")
        self.ax2.grid(True, which='both')
        self.ax2.legend()
        self.canvas2.draw()

        # ---------------------
        # Display text results
        # ---------------------
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "dt\tEuler\t\tHeun\t\tRK4\n")
        for i, dt in enumerate(dt_list):
            self.text.insert(tk.END, f"{dt:.3f}\t{errors['Euler'][i]:.3e}\t{errors['Heun'][i]:.3e}\t{errors['RK4'][i]:.3e}\n")

        v_target = 27.78
        idx = np.argmax(v_ref >= v_target)
        if v_ref[idx] >= v_target:
            self.text.insert(tk.END, f"\nReference time to reach 100 km/h ≈ {t_ref[idx]:.2f} s\n")
        else:
            self.text.insert(tk.END, "\nCar did not reach 100 km/h.\n")

# ==========================
# Run GUI
# ==========================
root = tk.Tk()
app = CarSimApp(root)
root.mainloop()
