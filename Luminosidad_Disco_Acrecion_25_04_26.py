# =============================================================================
# Maximización de la luminosidad de un disco de acreción
# Modelo: L(r,θ) = r² senθ (1+cosθ) exp(-r)   (con r0=1)
# Dominio: r ∈ [0.1, 5], θ ∈ [0, π/2]
# Punto inicial: r=0.5, θ=π/3 (≈60°)
# Método: ascenso por gradiente con paso adaptativo
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Función luminosidad
# ------------------------------------------------------------
def luminosidad(r, theta):
    """Calcula L(r,θ) = r^2 * sinθ * (1+cosθ) * exp(-r)"""
    return r**2 * np.sin(theta) * (1 + np.cos(theta)) * np.exp(-r)

# ------------------------------------------------------------
# 2. Gradiente numérico mediante diferencias finitas
# ------------------------------------------------------------
def gradiente_numerico(r, theta, delta=1e-5):
    """
    Aproxima el gradiente de L en (r,theta)
    devuelve (dL/dr, dL/dθ)
    """
    # Derivada respecto a r
    L_r_plus = luminosidad(r + delta, theta)
    L_r_minus = luminosidad(r - delta, theta)
    dL_dr = (L_r_plus - L_r_minus) / (2 * delta)

    # Derivada respecto a theta
    L_t_plus = luminosidad(r, theta + delta)
    L_t_minus = luminosidad(r, theta - delta)
    dL_dtheta = (L_t_plus - L_t_minus) / (2 * delta)

    return dL_dr, dL_dtheta

# ------------------------------------------------------------
# 3. Ascenso por gradiente con restricciones de dominio
# ------------------------------------------------------------
def ascenso_gradiente(r0, theta0, paso_inicial=0.05, tol=1e-7, max_iter=500):
    """
    Optimiza L mediante ascenso por gradiente.
    El paso se adapta automáticamente (backtracking simple) para evitar salidas del dominio.
    """
    r = r0
    theta = theta0
    historia = [(r, theta, luminosidad(r, theta))]

    paso = paso_inicial
    for i in range(max_iter):
        # Calcular gradiente
        dr, dtheta = gradiente_numerico(r, theta)
        grad_norm = np.sqrt(dr**2 + dtheta**2)

        if grad_norm < tol:
            break

        # Dirección de ascenso (normalizada)
        if grad_norm > 0:
            dr_norm = dr / grad_norm
            dtheta_norm = dtheta / grad_norm
        else:
            break

        # Prueba de paso (backtracking para no violar dominio)
        for _ in range(10):
            r_new = r + paso * dr_norm
            theta_new = theta + paso * dtheta_norm

            # Aplicar restricciones: r ∈ [0.1, 5], θ ∈ [0, π/2]
            r_new = np.clip(r_new, 0.1, 5.0)
            theta_new = np.clip(theta_new, 0.0, np.pi/2)

            # Si la función mejora, aceptamos el paso
            if luminosidad(r_new, theta_new) > luminosidad(r, theta):
                r, theta = r_new, theta_new
                historia.append((r, theta, luminosidad(r, theta)))
                break
            else:
                paso *= 0.5  # reducir paso
        else:
            # Si no se pudo mejorar, terminar
            break

        # Aumentar paso ligeramente para próxima iteración (opcional)
        paso = min(paso * 1.05, paso_inicial)

    return r, theta, luminosidad(r, theta), historia

# ------------------------------------------------------------
# 4. Parámetros y ejecución (caso c: r0=0.5, θ0=π/3)
# ------------------------------------------------------------
r_inicial = 0.5
theta_inicial = np.pi / 3   # 60°

r_optimo, theta_optimo, L_max, historial = ascenso_gradiente(r_inicial, theta_inicial)

print("=== OPTIMIZACIÓN DE L(r,θ) ===")
print(f"Punto de partida: r = {r_inicial:.3f}, θ = {theta_inicial:.3f} rad ({theta_inicial*180/np.pi:.1f}°)")
print(f"Punto óptimo encontrado: r* = {r_optimo:.4f}, θ* = {theta_optimo:.4f} rad ({theta_optimo*180/np.pi:.2f}°)")
print(f"Luminosidad máxima: L* = {L_max:.6f}")
print(f"Iteraciones realizadas: {len(historial)}")

# ------------------------------------------------------------
# 5. Gráfica de la función y la trayectoria de optimización
# ------------------------------------------------------------
# Malla para visualizar la superficie
r_vals = np.linspace(0.1, 5, 150)
theta_vals = np.linspace(0, np.pi/2, 150)
R, TH = np.meshgrid(r_vals, theta_vals)
L_vals = luminosidad(R, TH)

# Gráfica 2D con contornos y trayectoria
plt.figure(figsize=(10, 8))
contour = plt.contourf(R, TH, L_vals, levels=30, cmap='plasma')
plt.colorbar(contour, label='Luminosidad L(r,θ)')
plt.clim(0, np.max(L_vals))

# Trayectoria del gradiente
hist_r = [p[0] for p in historial]
hist_theta = [p[1] for p in historial]
plt.plot(hist_r, hist_theta, 'w-o', linewidth=2, markersize=4, label='Trayectoria de ascenso')
plt.plot(r_inicial, theta_inicial, 'go', markersize=8, label='Punto inicial')
plt.plot(r_optimo, theta_optimo, 'r*', markersize=12, label='Óptimo encontrado')

plt.xlabel('Radio r (unidades r₀)', fontsize=12)
plt.ylabel('Ángulo θ (radianes)', fontsize=12)
plt.title('Maximización de la luminosidad del disco de acreción\n(caso: r₀=0.5, θ₀=π/3)', fontsize=13)
plt.xlim(0.1, 5)
plt.ylim(0, np.pi/2)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Gráfica de la evolución de la luminosidad (convergencia)
# ------------------------------------------------------------
# Extraer las luminosidades del historial y crear el vector de iteraciones
L_hist = [p[2] for p in historial]
iteraciones = list(range(len(L_hist)))
diferencia = [L_hist[0] - L for L in L_hist]

plt.figure(figsize=(9, 5))
plt.plot(iteraciones, diferencia, 'b-', linewidth=1.5, label='Diferencia con el máximo final')
plt.xlabel('Iteración', fontsize=12)
plt.ylabel('L(máx) - L(iteración)', fontsize=12)
plt.title('Convergencia del ascenso por gradiente', fontsize=13)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()