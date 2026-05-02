# ============================================================================
# PARTE 1: Suma de Riemann por la derecha para ∫₀ᵇ x³ dx
# ============================================================================

import numpy as np

def riemann_derecha(funcion, limite_sup, num_subintervalos):
    delta = limite_sup / num_subintervalos
    indices = np.arange(1, num_subintervalos + 1)   # extremos derechos
    puntos = indices * delta
    return np.sum(funcion(puntos)) * delta

def cubo(x):
    return x ** 3

b = 2.0
n = 3

estimacion = riemann_derecha(cubo, b, n)
exacto = b**4 / 4

print("=== Suma de Riemann (extremos derechos) ===")
print(f"n = {n}, ∫₀^{b} x³ dx ≈ {estimacion}")
print(f"Valor exacto = {exacto}")

error_abs = abs(exacto - estimacion)
error_rel = error_abs / exacto

print(f"Error absoluto = {error_abs}")
print(f"Error relativo = {error_rel}")
print(f"Error relativo (%) = {error_rel*100:.6f}\n")

# ============================================================================
# PARTE 2: Regla del trapecio compuesta para una integral de cohete
# ============================================================================

from scipy.integrate import quad

def velocidad_cohete(t):
    """Velocidad del cohete (modelo del problema)."""
    return 2000 * np.log(140000 / (140000 - 2100 * t)) - 9.8 * t

t_inicio = 8.0
t_fin = 30.0
n_trapecios = 3

delta_t = (t_fin - t_inicio) / n_trapecios
nodos = np.linspace(t_inicio, t_fin, n_trapecios + 1)
valores = velocidad_cohete(nodos)

# Trapecio compuesto
integral_trap = (delta_t / 2) * (valores[0] + 2 * np.sum(valores[1:-1]) + valores[-1])

# Valor de referencia (cuadratura adaptativa)
integral_ref, _ = quad(velocidad_cohete, t_inicio, t_fin)

error_abs2 = abs(integral_ref - integral_trap)
error_rel2 = error_abs2 / abs(integral_ref)

print("=== Regla del trapecio compuesta (n=3) ===")
print(f"h = {delta_t}")
print(f"Nodos t = {nodos}")
print(f"f(t) = {valores}")
print(f"\nT₃ ≈ {integral_trap}")
print(f"Referencia (quad) ≈ {integral_ref}")
print(f"Error absoluto = {error_abs2}")
print(f"Error relativo = {error_rel2}")
print(f"Error relativo (%) = {error_rel2*100:.6f}\n")

# ============================================================================
# PARTE 3: Estudio de convergencia (n = 1 a 8) con tabla en texto
# ============================================================================

import pandas as pd

def trapecio_adaptable(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    T = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return T

# Valor real ya calculado (integral_ref)
resultados = []
for m in range(1, 9):
    T_m = trapecio_adaptable(velocidad_cohete, t_inicio, t_fin, m)
    err_abs_m = abs(integral_ref - T_m)
    err_rel_m = err_abs_m / abs(integral_ref)
    resultados.append([m, T_m, err_abs_m, err_rel_m, err_rel_m * 100])

tabla_convergencia = pd.DataFrame(
    resultados,
    columns=[
        "Subintervalos (n)",
        "Aprox. trapecio",
        "Error absoluto",
        "Error relativo",
        "Error relativo (%)"
    ]
)

# Mostrar la tabla sin usar IPython
print("=== Convergencia de la regla del trapecio compuesta ===")
print(tabla_convergencia.to_string(index=False, float_format="%.4f"))