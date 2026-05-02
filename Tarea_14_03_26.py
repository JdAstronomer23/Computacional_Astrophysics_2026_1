# ============================================================================
# EJERCICIO 1: Newton-Raphson para sistema no lineal 2x2
# ============================================================================

import numpy as np

def sistema_ecuaciones(u, v):
    """Sistema F(u,v) = [u*v - 2, u^2 + v - 5]^T"""
    return np.array([
        u * v - 2,
        u**2 + v - 5
    ], dtype=float)

def jacobiano(u, v):
    """Matriz Jacobiana del sistema."""
    return np.array([
        [v, u],
        [2*u, 1]
    ], dtype=float)

# Semilla inicial
u_actual = 3.0
v_actual = 4.0

tolerancia = 0.05
max_iteraciones = 100

print("Iter |      u        |      v        |   err_u     |   err_v")
print("-" * 65)

for contador in range(max_iteraciones):
    F_val = sistema_ecuaciones(u_actual, v_actual)
    J_val = jacobiano(u_actual, v_actual)
    
    # Resolver sistema J * delta = -F
    delta = np.linalg.solve(J_val, -F_val)
    
    u_nuevo = u_actual + delta[0]
    v_nuevo = v_actual + delta[1]
    
    # Errores relativos (con protección contra división por cero)
    err_u = abs((u_nuevo - u_actual) / u_nuevo) if u_nuevo != 0 else abs(u_nuevo - u_actual)
    err_v = abs((v_nuevo - v_actual) / v_nuevo) if v_nuevo != 0 else abs(v_nuevo - v_actual)
    
    print(f"{contador+1:4d} | {u_nuevo:10.6f} | {v_nuevo:10.6f} | {err_u:10.6f} | {err_v:10.6f}")
    
    # Actualizar para siguiente iteración
    u_actual, v_actual = u_nuevo, v_nuevo
    
    if err_u < tolerancia and err_v < tolerancia:
        break

print("\n=== Solución aproximada del sistema ===")
print(f"u ≈ {u_actual:.6f}")
print(f"v ≈ {v_actual:.6f}\n")

# ============================================================================
# EJERCICIO 2: Método de la secante para encontrar raíz de una cúbica
# ============================================================================

def polinomio(x):
    """f(x) = x^3 - 0.165 x^2 + 3.993e-4"""
    return x**3 - 0.165 * x**2 + 3.993e-4

# Dos aproximaciones iniciales
x_prev = 2.0   # x_{n-1}
x_curr = 3.0   # x_n

print("\n=== Método de la secante (3 iteraciones) ===")
print("Iter.   x_nuevo          error_relativo")

for paso in range(1, 4):
    # Fórmula de la secante
    x_next = x_curr - polinomio(x_curr) * (x_curr - x_prev) / (polinomio(x_curr) - polinomio(x_prev))
    
    # Error relativo
    error = abs((x_next - x_curr) / x_next) if x_next != 0 else abs(x_next - x_curr)
    
    print(f"{paso:4d}   {x_next:12.8f}   {error:12.8f}")
    
    # Desplazar los valores
    x_prev, x_curr = x_curr, x_next

# ============================================================================
# EJERCICIO 3: Método de Newton-Raphson escalar (misma función)
# ============================================================================

def derivada_polinomio(x):
    """f'(x) = 3x^2 - 0.33x"""
    return 3 * x**2 - 0.33 * x

# Semilla
x0_newton = 0.05
iteraciones = 3

print("\n=== Método de Newton-Raphson (3 iteraciones) ===")
print("Iter.   x_nuevo          error_relativo")

x_ant = x0_newton
for paso in range(1, iteraciones + 1):
    x_nue = x_ant - polinomio(x_ant) / derivada_polinomio(x_ant)
    error_rel = abs((x_nue - x_ant) / x_nue) if x_nue != 0 else abs(x_nue - x_ant)
    
    print(f"{paso:4d}   {x_nue:12.10f}   {error_rel:12.10f}")
    
    x_ant = x_nue