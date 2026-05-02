import math

def area(theta):
    """Área de la sección transversal: A = 4 sinθ (1+cosθ)"""
    return 4 * math.sin(theta) * (1 + math.cos(theta))

def d_area(theta):
    """Primera derivada de A(θ): A'(θ) = 4 cosθ + 4 cos(2θ)"""
    return 4 * math.cos(theta) + 4 * math.cos(2 * theta)

def d2_area(theta):
    """Segunda derivada de A(θ): A''(θ) = -4 sinθ - 8 sin(2θ)"""
    return -4 * math.sin(theta) - 8 * math.sin(2 * theta)

def newton_maximization(f, df, d2f, x0, tol=1e-6, max_iter=100):
    """
    Encuentra el máximo de f(x) usando el método de Newton.
    Iteración: x_{n+1} = x_n - df(x_n) / d2f(x_n)
    
    Parámetros:
    f : función objetivo
    df : primera derivada
    d2f : segunda derivada (debe ser negativa en la vecindad del máximo)
    x0 : valor inicial
    tol : tolerancia para el cambio relativo en x
    max_iter : número máximo de iteraciones
    """
    x = x0
    print("Iter |       x_n       |      f(x_n)      |       df(x_n)     |      d2f(x_n)     |    Cambio")
    print("-" * 85)
    
    for i in range(max_iter):
        f_val = f(x)
        df_val = df(x)
        d2f_val = d2f(x)
        
        # Verificar si d2f es cercano a cero
        if abs(d2f_val) < 1e-12:
            print(f"Advertencia: segunda derivada casi cero en x={x:.6f}. Deteniendo.")
            break
        
        # Actualización de Newton
        x_new = x - df_val / d2f_val
        cambio = x_new - x
        
        print(f"{i+1:4d} | {x:15.8f} | {f_val:15.8f} | {df_val:15.8f} | {d2f_val:15.8f} | {cambio:12.6e}")
        
        # Criterio de parada (cambio absoluto pequeño)
        if abs(cambio) < tol:
            print(f"\nConvergencia alcanzada después de {i+1} iteraciones.")
            break
        
        x = x_new
    else:
        print(f"\nMáximo de iteraciones ({max_iter}) alcanzado.")
    
    return x, f(x)

# Parámetros del problema
x0 = 1.0  # valor inicial en radianes (dentro del intervalo [0, π/2])
tol = 1e-6

# Ejecutar método de Newton
theta_opt_newton, area_opt_newton = newton_maximization(area, d_area, d2_area, x0, tol)

# Solución exacta
theta_exact = math.pi / 3   # 60° = 1.0471975512 rad
area_exact = area(theta_exact)

# Errores
error_abs_theta = abs(theta_opt_newton - theta_exact)
error_rel_theta = error_abs_theta / theta_exact
error_abs_area = abs(area_opt_newton - area_exact)
error_rel_area = error_abs_area / area_exact

print("\n--- Resultados del método de Newton ---")
print(f"θ óptimo encontrado: {theta_opt_newton:.10f} rad")
print(f"A máxima encontrada: {area_opt_newton:.10f}")
print(f"Solución exacta: θ* = {theta_exact:.10f} rad, A* = {area_exact:.10f}")
print(f"Error absoluto en θ: {error_abs_theta:.2e} rad")
print(f"Error relativo en θ: {error_rel_theta:.6%}")
print(f"Error absoluto en A: {error_abs_area:.2e}")
print(f"Error relativo en A: {error_rel_area:.6%}")

# Verificar que la segunda derivada sea negativa en el óptimo
print(f"\nSegunda derivada en el óptimo: d2A/dθ² = {d2_area(theta_opt_newton):.6f} (debe ser negativa para máximo)")