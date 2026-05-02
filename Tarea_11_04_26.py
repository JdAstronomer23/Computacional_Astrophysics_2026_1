# ============================================================================
# Simulación numérica de la evolución de energía en una erupción solar
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Parámetros físicos y numéricos
# ------------------------------------------------------------
indice_no_lineal = 1.5        # exponente n
energia_inicial = 1.0         # E0
instante_inicial = 0.0
instante_final = 10.0
paso_temporal = 0.01
coeficiente_alfa = 1.0        # α (constante de decaimiento)

# ------------------------------------------------------------
# 2. Ecuación diferencial dE/dt = -α E^n
# ------------------------------------------------------------
def tasa_cambio(t, E):
    return -coeficiente_alfa * (E ** indice_no_lineal)

# ------------------------------------------------------------
# 3. Integrador Runge-Kutta de orden 4 (RK4)
# ------------------------------------------------------------
def integrador_rk4(ecuacion, t_i, t_f, y_i, paso):
    tiempos = np.arange(t_i, t_f + paso, paso)
    solucion = np.zeros(len(tiempos))
    solucion[0] = y_i

    for j in range(len(tiempos) - 1):
        t_act = tiempos[j]
        y_act = solucion[j]

        k1 = ecuacion(t_act, y_act)
        k2 = ecuacion(t_act + paso/2, y_act + (paso/2)*k1)
        k3 = ecuacion(t_act + paso/2, y_act + (paso/2)*k2)
        k4 = ecuacion(t_act + paso, y_act + paso*k3)

        solucion[j+1] = y_act + (paso/6)*(k1 + 2*k2 + 2*k3 + k4)

    return tiempos, solucion

# Resolver la ODE
tiempos_rk, energia_rk = integrador_rk4(
    tasa_cambio,
    instante_inicial,
    instante_final,
    energia_inicial,
    paso_temporal
)

# ------------------------------------------------------------
# 4. Interpolación lineal para evaluar E(t) en cualquier t
# ------------------------------------------------------------
def energia_interpolada(t):
    return np.interp(t, tiempos_rk, energia_rk)

# ------------------------------------------------------------
# 5. Función cuya raíz buscamos: g(t) = E(t) - E0/2
# ------------------------------------------------------------
def funcion_objetivo(t):
    return energia_interpolada(t) - 0.5

# Derivada numérica mediante diferencias centradas
def derivada_numerica(func, t, delta=1e-5):
    return (func(t + delta) - func(t - delta)) / (2.0 * delta)

# ------------------------------------------------------------
# 6. Método de Newton-Raphson para hallar t tal que E(t)=0.5
# ------------------------------------------------------------
def newton_raphson(semilla, tolerancia=1e-10, max_iter=50):
    x = semilla
    historial = []

    for iteracion in range(max_iter):
        g_val = funcion_objetivo(x)
        dg_val = derivada_numerica(funcion_objetivo, x)

        historial.append([iteracion + 1, x, g_val, dg_val])

        if abs(g_val) < tolerancia:
            return x, historial

        if abs(dg_val) < 1e-14:
            raise RuntimeError("Derivada casi nula – método falla.")

        x = x - g_val / dg_val

    raise RuntimeError("Newton-Raphson no converge en el número máximo de iteraciones.")

# Semilla inicial (tiempo más cercano a E=0.5 en la malla de RK4)
idx_cercano = np.argmin(np.abs(energia_rk - 0.5))
semilla_t = tiempos_rk[idx_cercano]

t_media_vida, registro_newton = newton_raphson(semilla_t)

# ------------------------------------------------------------
# 7. Mostrar resultados en formato tabla (sin IPython)
# ------------------------------------------------------------
df_newton = pd.DataFrame(registro_newton, columns=[
    "Iter", "t (s)", "g(t)=E(t)-0.5", "g'(t)"
])

print("\n=== Progreso del método de Newton-Raphson ===")
print(df_newton.to_string(
    index=False,
    formatters={
        "t (s)": "{:.8f}".format,
        "g(t)=E(t)-0.5": "{:.6e}".format,
        "g'(t)": "{:.6e}".format
    }
))

print(f"\n>>> Tiempo de vida media tm = {t_media_vida:.8f} s")
print(f">>> Energía en ese instante E(tm) = {energia_interpolada(t_media_vida):.8f}")

# ------------------------------------------------------------
# 8. Visualización
# ------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.plot(tiempos_rk, energia_rk, 'b-', linewidth=2, label='E(t) por RK4')
plt.axhline(y=0.5, color='r', linestyle='--', label='E₀/2 = 0.5')
plt.axvline(x=t_media_vida, color='g', linestyle='--', label=f'tm ≈ {t_media_vida:.4f} s')
plt.xlabel('Tiempo (s)')
plt.ylabel('Energía E(t)')
plt.title('Decaimiento de energía en una erupción solar (RK4 + Newton)')
plt.grid(alpha=0.4, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()