# =============================================================================
# Pulsación radial amortiguada de una estrella
# 
# Ecuación: d²x/dt² + ω₀ ξ dx/dt + ω₀² x = 0
# 
# Transformación a sistema de primer orden:
#   x₁ = x
#   x₂ = dx/dt
#   dx₁/dt = x₂
#   dx₂/dt = -ω₀² x₁ - ω₀ ξ x₂
#
# Condiciones iniciales:
#   x(0) = 0.01 * R₀
#   x'(0) = 0
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Parámetros físicos
# ------------------------------------------------------------
R0 = 1.0                # radio medio de la estrella (unidades arbitrarias)
omega0 = 1.0            # frecuencia natural de pulsación (rad/s)
xi = 0.2                # parámetro de amortiguamiento (adimensional)

# Condiciones iniciales (según el problema)
x0 = 0.01 * R0          # desplazamiento inicial pequeño
v0 = 0.0                # velocidad radial inicial nula

# Parámetros numéricos
t_inicial = 0.0
t_final = 30.0          # tiempo suficiente para ver varias oscilaciones
h = 0.01                # paso de integración (RK4)

# ------------------------------------------------------------
# 1. Definición del sistema de EDOs de primer orden
# ------------------------------------------------------------
def sistema_estrella(t, estado):
    """
    estado = [x, v]
    dxdt = v
    dvdt = -omega0^2 * x - omega0 * xi * v
    """
    x, v = estado
    dxdt = v
    dvdt = -omega0**2 * x - omega0 * xi * v
    return np.array([dxdt, dvdt])

# ------------------------------------------------------------
# 2. Implementación del método RK4
# ------------------------------------------------------------
def rk4(derivs, t0, tf, y0, h):
    n_pasos = int((tf - t0) / h) + 1
    t_vals = np.linspace(t0, tf, n_pasos)
    y_vals = np.zeros((n_pasos, len(y0)))
    y_vals[0] = y0

    for i in range(n_pasos - 1):
        t = t_vals[i]
        y = y_vals[i]

        k1 = derivs(t, y)
        k2 = derivs(t + h/2, y + (h/2)*k1)
        k3 = derivs(t + h/2, y + (h/2)*k2)
        k4 = derivs(t + h, y + h*k3)

        y_vals[i+1] = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t_vals, y_vals

# ------------------------------------------------------------
# 3. Resolución numérica
# ------------------------------------------------------------
t, sol = rk4(sistema_estrella, t_inicial, t_final, [x0, v0], h)
x_num = sol[:, 0]
v_num = sol[:, 1]

# ------------------------------------------------------------
# 4. Gráficas (estilo publicación científica)
# ------------------------------------------------------------
plt.figure(figsize=(12, 5))

# Subplot 1: Desplazamiento x(t)
plt.subplot(1, 2, 1)
plt.plot(t, x_num, 'b-', linewidth=1.5, label=r'$x(t) = R(t)-R_0$')
plt.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
plt.xlabel('Tiempo (s)', fontsize=11)
plt.ylabel(r'$x(t)$ (en unidades de $R_0$)', fontsize=11)
plt.title('Desplazamiento radial (pulsación amortiguada)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()

# Subplot 2: Velocidad radial v(t)
plt.subplot(1, 2, 2)
plt.plot(t, v_num, 'r-', linewidth=1.5, label=r'$v(t) = dx/dt$')
plt.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
plt.xlabel('Tiempo (s)', fontsize=11)
plt.ylabel(r'$v(t)$ (unidades de $R_0$/s)', fontsize=11)
plt.title('Velocidad radial', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. Información adicional y comprobación de amortiguamiento
# ------------------------------------------------------------
# Cálculo del decremento logarítmico (opcional)
# Busco picos positivos cerca del inicio y después de algunos ciclos
from scipy.signal import find_peaks

picos, _ = find_peaks(x_num, height=0)  # picos positivos
if len(picos) >= 2:
    amp1 = x_num[picos[0]]
    amp2 = x_num[picos[-1]]
    ciclos = len(picos) - 1
    decremento = (1/ciclos) * np.log(amp1/amp2)
    print(f"\nDecremento logarítmico aproximado: δ = {decremento:.4f}")
    print(f"Amortiguamiento teórico (ξ = {xi}) -> δ_teórico = {np.pi*xi/np.sqrt(1-xi**2):.4f} (aprox)")

print(f"\nParámetros utilizados:")
print(f"   ω₀ = {omega0} rad/s")
print(f"   ξ  = {xi}")
print(f"   R₀ = {R0} (unidades arbitrarias)")
print(f"   x(0) = {x0},  x'(0) = {v0}")
print(f"   Paso RK4: h = {h}")