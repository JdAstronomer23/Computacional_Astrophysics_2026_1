# =============================================================================
# Optimización de la temperatura de emisión estelar
# Método de la Sección Áurea (Golden Section Search)
#
# Modelo: P(T) = σ T^4 exp(-T/T0) [1 - exp(-hν/(kB T))]
# con σ = 5.67e-8, T0 = 10000 K, hν/kB = 5000 K
# Intervalo: T ∈ [3000, 50000] K, tolerancia ε = 50 K
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Parámetros físicos y función de potencia
# ------------------------------------------------------------
sigma = 5.67e-8   # W m^{-2} K^{-4}
T0 = 10000.0      # K
hnu_kB = 5000.0   # K (equivalente a hν/k_B)

def potencia_estelar(T):
    """
    Calcula la potencia emitida por unidad de área (W/m^2)
    para una temperatura dada T (en Kelvin).
    """
    factor1 = sigma * (T**4)
    factor2 = np.exp(-T / T0)
    factor3 = 1.0 - np.exp(-hnu_kB / T)
    return factor1 * factor2 * factor3

# ------------------------------------------------------------
# 2. Implementación del método de la sección áurea (maximización)
# ------------------------------------------------------------
def golden_section_max(f, a, b, tol=50.0, max_iter=100):
    """
    Encuentra el máximo de la función f en el intervalo [a, b]
    utilizando el método de la sección áurea.
    
    Parámetros:
        f   : función objetivo (a maximizar)
        a,b : límites iniciales del intervalo
        tol : tolerancia en la longitud del intervalo final
        max_iter : número máximo de iteraciones
    
    Retorna:
        x_max : valor que maximiza f (dentro de la tolerancia)
        f_max : valor máximo de f
        iteraciones realizadas
    """
    # Proporción áurea y su complemento
    gr = (np.sqrt(5) - 1) / 2   # ≈ 0.618034 (inverso de φ)
    # Inicializar puntos interiores
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    
    iteracion = 0
    while (b - a) > tol and iteracion < max_iter:
        if f1 > f2:               # máximo en el subintervalo [a, x2]
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - gr * (b - a)
            f1 = f(x1)
        else:                      # máximo en [x1, b]
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + gr * (b - a)
            f2 = f(x2)
        iteracion += 1
    
    # El punto óptimo es el punto medio del intervalo final
    x_opt = (a + b) / 2
    f_opt = f(x_opt)
    return x_opt, f_opt, iteracion

# ------------------------------------------------------------
# 3. Búsqueda del máximo en el rango de interés
# ------------------------------------------------------------
T_min = 3000.0   # K
T_max = 50000.0  # K
epsilon = 50.0   # K (precisión requerida)

T_optimo, P_max, n_iters = golden_section_max(potencia_estelar, T_min, T_max, tol=epsilon)

print("=== RESULTADOS DE LA OPTIMIZACIÓN ===")
print(f"Método: Sección Áurea")
print(f"Intervalo inicial: [{T_min:.0f}, {T_max:.0f}] K")
print(f"Tolerancia (ΔT): {epsilon:.1f} K")
print(f"Iteraciones realizadas: {n_iters}")
print(f"\nTemperatura óptima: T* = {T_optimo:.2f} K")
print(f"Potencia máxima: P(T*) = {P_max:.3e} W/m²")

# ------------------------------------------------------------
# 4. Gráfica de P(T) con indicación del máximo
# ------------------------------------------------------------
# Generar valores para la curva (resolución fina)
T_graf = np.linspace(T_min, T_max, 500)
P_graf = potencia_estelar(T_graf)

plt.figure(figsize=(9, 6))
plt.plot(T_graf, P_graf, 'b-', linewidth=2, label=r'$P(T) = \sigma T^4 e^{-T/T_0} (1 - e^{-h\nu/k_B T})$')
plt.plot(T_optimo, P_max, 'ro', markersize=10, label=f'Máximo (T* = {T_optimo:.1f} K)')
plt.axvline(T_optimo, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Temperatura (K)', fontsize=12)
plt.ylabel('Potencia emitida (W/m²)', fontsize=12)
plt.title('Optimización de la emisión estelar - Método de la Sección Áurea', fontsize=13)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()