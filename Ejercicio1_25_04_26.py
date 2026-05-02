import math

def golden_section_max(f, x_l, x_u, epsilon):
    """
    Encuentra el máximo de f(x) en el intervalo [x_l, x_u]
    usando el método de la Sección Áurea.
    
    Parámetros:
    f : función a maximizar
    x_l, x_u : límites del intervalo
    epsilon : tolerancia para el ancho del intervalo
    
    Retorna:
    x_opt : punto donde se estima el máximo
    f_opt : valor de la función en x_opt
    iteraciones : número de iteraciones realizadas
    """
    # Constante de la sección áurea (R = (sqrt(5)-1)/2 ≈ 0.618034)
    R = (math.sqrt(5) - 1) / 2
    
    # Inicialización
    iter_count = 0
    
    # Calcular los dos puntos interiores
    d = R * (x_u - x_l)
    x1 = x_l + d
    x2 = x_u - d
    
    f1 = f(x1)
    f2 = f(x2)
    
    print("Iter |       x_l       |       x_u       |       x1       |       x2       |      f(x1)     |      f(x2)     |   Ancho")
    print("-" * 100)
    
    while (x_u - x_l) > epsilon:
        iter_count += 1
        print(f"{iter_count:4d} | {x_l:15.6f} | {x_u:15.6f} | {x1:15.6f} | {x2:15.6f} | {f1:15.6f} | {f2:15.6f} | {(x_u-x_l):12.6f}")
        
        if f1 > f2:  # El máximo está en [x_l, x1]
            x_u = x1
            x1 = x2          # El nuevo x1 será el anterior x2
            f1 = f2
            # Recalcular x2 y f2 para el nuevo intervalo
            x2 = x_u - R * (x_u - x_l)
            f2 = f(x2)
        else:  # f1 < f2: el máximo está en [x2, x_u]
            x_l = x2
            x2 = x1          # El nuevo x2 será el anterior x1
            f2 = f1
            # Recalcular x1 y f1 para el nuevo intervalo
            x1 = x_l + R * (x_u - x_l)
            f1 = f(x1)
    
    # Estimación final del máximo
    x_opt = (x_l + x_u) / 2
    f_opt = f(x_opt)
    
    print("-" * 100)
    print(f"\nIntervalo final: [{x_l:.6f}, {x_u:.6f}], ancho = {x_u - x_l:.6f} < ε = {epsilon}")
    print(f"Punto óptimo estimado: θ = {x_opt:.6f} rad")
    print(f"Valor máximo estimado: A = {f_opt:.6f}")
    print(f"Iteraciones realizadas: {iter_count}")
    
    return x_opt, f_opt, iter_count

# Definición de la función a maximizar
def area(theta):
    """Área de la sección transversal del canal: A = 4 sinθ (1 + cosθ)"""
    return 4 * math.sin(theta) * (1 + math.cos(theta))

# Parámetros del problema
x_l = 0.0
x_u = math.pi / 2   # [0, π/2]
epsilon = 0.05

# Ejecutar el método de la sección áurea
theta_opt, area_opt, iters = golden_section_max(area, x_l, x_u, epsilon)

# Solución exacta
theta_exact = math.pi / 3   # 1.0471975512 rad -> 60°
area_exact = 4 * math.sin(theta_exact) * (1 + math.cos(theta_exact))

# Errores absolutos
error_abs_theta = abs(theta_opt - theta_exact)
error_abs_area = abs(area_opt - area_exact)

# Errores relativos
if theta_exact != 0:
    error_rel_theta = error_abs_theta / theta_exact
else:
    error_rel_theta = float('inf')

if area_exact != 0:
    error_rel_area = error_abs_area / area_exact
else:
    error_rel_area = float('inf')

print(f"\nθ óptimo (valor numérico) = {theta_opt:.6f} rad")
print(f"A óptima (valor numérico) = {area_opt:.6f}")

print("\n--- Comparación con la solución exacta ---")
print(f"Solución exacta: θ* = {theta_exact:.6f} rad, A* = {area_exact:.6f}")
print(f"Error absoluto en θ:     {error_abs_theta:.6f} rad")
print(f"Error absoluto en A:     {error_abs_area:.6f}")
print(f"Error relativo en θ:     {error_rel_theta:.6%}")
print(f"Error relativo en A:     {error_rel_area:.6%}")