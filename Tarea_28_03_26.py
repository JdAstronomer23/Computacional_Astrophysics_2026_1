# ============================================================================
# Ajuste exponencial por mínimos cuadrados (método de Lomb-Scargle simplificado)
# ============================================================================

import numpy as np

# ------------------------------------------------------------
# 1. Datos de decaimiento radiactivo
# ------------------------------------------------------------
horas = np.array([0, 1, 3, 5, 7, 9], dtype=float)
intensidad = np.array([1.0, 0.891, 0.708, 0.562, 0.447, 0.355], dtype=float)

# ------------------------------------------------------------
# 2. Función objetivo para encontrar λ (ecuación de punto crítico)
# ------------------------------------------------------------
def ecuacion_tasa_decaimiento(lam):
    """
    f(λ) = Σ(γ_i t_i e^{λ t_i}) - 
           [Σ(γ_i e^{λ t_i}) / Σ(e^{2λ t_i})] * Σ(t_i e^{2λ t_i})
    """
    num1 = np.sum(intensidad * horas * np.exp(lam * horas))
    
    suma_γ_e = np.sum(intensidad * np.exp(lam * horas))
    suma_e2 = np.sum(np.exp(2 * lam * horas))
    factor = suma_γ_e / suma_e2
    suma_t_e2 = np.sum(horas * np.exp(2 * lam * horas))
    
    return num1 - factor * suma_t_e2

# ------------------------------------------------------------
# 3. Método de búsqueda de raíz (bisección)
# ------------------------------------------------------------
def buscar_raiz_biseccion(func, lim_inf, lim_sup, tolerancia=1e-8, max_iters=1000):
    """
    Encuentra la raíz de func en [lim_inf, lim_sup] por bisección.
    """
    if func(lim_inf) * func(lim_sup) > 0:
        raise ValueError("El intervalo no contiene una raíz (cambio de signo).")
    
    contador = 0
    while (lim_sup - lim_inf) > tolerancia and contador < max_iters:
        punto_medio = (lim_inf + lim_sup) / 2
        
        if func(lim_inf) * func(punto_medio) < 0:
            lim_sup = punto_medio
        else:
            lim_inf = punto_medio
        
        contador += 1
    
    raiz = (lim_inf + lim_sup) / 2
    return raiz, contador

# ------------------------------------------------------------
# 4. Construcción del intervalo según los últimos dos dígitos de cédula
#    (usamos nc = 49)
# ------------------------------------------------------------
valor_aproximado_lambda = -0.11505
ultimos_dos = 49          # último par de dígitos de la cédula
desplazamiento = ultimos_dos / 1000  # 0.049

limite_inferior = valor_aproximado_lambda - desplazamiento
limite_superior = valor_aproximado_lambda + desplazamiento

print("=== Intervalo de bisección ===")
print(f"Límite inferior λ_min = {limite_inferior:.8f}")
print(f"Límite superior λ_max = {limite_superior:.8f}")

# ------------------------------------------------------------
# 5. Cálculo de λ por bisección
# ------------------------------------------------------------
lambda_optimo, num_iter = buscar_raiz_biseccion(
    ecuacion_tasa_decaimiento,
    limite_inferior,
    limite_superior,
    tolerancia=1e-8
)

print("\n=== Resultado de la bisección ===")
print(f"λ óptimo = {lambda_optimo:.8f}")
print(f"Iteraciones realizadas = {num_iter}")

# ------------------------------------------------------------
# 6. Cálculo de la amplitud A
# ------------------------------------------------------------
A_optimo = (np.sum(intensidad * np.exp(lambda_optimo * horas)) /
            np.sum(np.exp(2 * lambda_optimo * horas)))

print(f"\nAmplitud A = {A_optimo:.8f}")

# ------------------------------------------------------------
# 7. Vida media (half-life)
# ------------------------------------------------------------
vida_media = np.log(2) / abs(lambda_optimo)
print(f"\nVida media = {vida_media:.8f} horas")

# ------------------------------------------------------------
# 8. Intensidad predicha después de 24 horas
# ------------------------------------------------------------
intensidad_24h = A_optimo * np.exp(lambda_optimo * 24)
print(f"\nIntensidad a las 24 horas = {intensidad_24h:.8f}")