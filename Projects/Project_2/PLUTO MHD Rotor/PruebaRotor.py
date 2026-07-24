import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. Parámetros Globales y Malla
# ==========================================
Nx, Ny = 200, 200  # Resolución (ajustable según capacidad de cómputo)
x = np.linspace(-0.5, 0.5, Nx)
y = np.linspace(-0.5, 0.5, Ny)
X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]
gamma = 1.4
CFL = 0.4
t_end = 0.15

# ==========================================
# 2. Funciones de Ecuación de Estado (EOS)
# ==========================================
def get_primitives(U):
    """Convierte variables conservadas a primitivas."""
    rho = U[0]
    vx = U[1] / rho
    vy = U[2] / rho
    vz = U[3] / rho
    Bx, By, Bz = U[5], U[6], U[7]
    
    # Energía cinética y magnética
    E_k = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    E_b = 0.5 * (Bx**2 + By**2 + Bz**2)
    
    # Presión térmica (p = (gamma-1)*(E - Ek - Eb))
    p = (gamma - 1.0) * (U[4] - E_k - E_b)
    
    # Evitar presiones negativas por errores numéricos
    p = np.maximum(p, 1e-9)
    
    return rho, vx, vy, vz, p, Bx, By, Bz

def compute_fast_magnetosonic(rho, vx, vy, vz, p, Bx, By, Bz):
    """Calcula la velocidad magnetosónica rápida máxima."""
    a2 = gamma * p / rho
    b2 = (Bx**2 + By**2 + Bz**2) / rho
    # Velocidad máxima de onda en direcciones X e Y
    cf_x = np.sqrt(0.5 * (a2 + b2 + np.sqrt((a2 + b2)**2 - 4 * a2 * (Bx**2/rho))))
    cf_y = np.sqrt(0.5 * (a2 + b2 + np.sqrt((a2 + b2)**2 - 4 * a2 * (By**2/rho))))
    return cf_x, cf_y

# ==========================================
# 3. Cálculo de Flujos (Solver de Rusanov / LLF)
# ==========================================
def compute_fluxes(U):
    """Calcula los flujos F(U) en X y G(U) en Y."""
    rho, vx, vy, vz, p, Bx, By, Bz = get_primitives(U)
    
    p_star = p + 0.5 * (Bx**2 + By**2 + Bz**2)
    v_dot_B = vx*Bx + vy*By + vz*Bz
    
    # Flujos F en dirección X
    F = np.zeros_like(U)
    F[0] = rho * vx
    F[1] = rho * vx**2 + p_star - Bx**2
    F[2] = rho * vx * vy - Bx * By
    F[3] = rho * vx * vz - Bx * Bz
    F[4] = (U[4] + p_star) * vx - Bx * v_dot_B
    F[5] = np.zeros_like(Bx) # div(B)=0 control (Bx constante en 1D)
    F[6] = vx * By - vy * Bx
    F[7] = vx * Bz - vz * Bx
    
    # Flujos G en dirección Y
    G = np.zeros_like(U)
    G[0] = rho * vy
    G[1] = rho * vy * vx - By * Bx
    G[2] = rho * vy**2 + p_star - By**2
    G[3] = rho * vy * vz - By * Bz
    G[4] = (U[4] + p_star) * vy - By * v_dot_B
    G[5] = vy * Bx - vx * By
    G[6] = np.zeros_like(By) # div(B)=0 control
    G[7] = vy * Bz - vz * By
    
    return F, G, rho, vx, vy, vz, p, Bx, By, Bz

def rusanov_update(U, dx, dy, dt):
    """Actualización espacial usando flujos numéricos de Lax-Friedrichs."""
    F, G, rho, vx, vy, vz, p, Bx, By, Bz = compute_fluxes(U)
    cf_x, cf_y = compute_fast_magnetosonic(rho, vx, vy, vz, p, Bx, By, Bz)
    
    # Velocidades máximas en la malla para difusión numérica
    cx_max = np.max(np.abs(vx) + cf_x)
    cy_max = np.max(np.abs(vy) + cf_y)
    
    # Desplazamientos de malla (Condiciones periódicas)
    U_R = np.roll(U, -1, axis=2) # U_{i+1, j}
    U_L = np.roll(U, 1, axis=2)  # U_{i-1, j}
    U_U = np.roll(U, -1, axis=1) # U_{i, j+1}
    U_D = np.roll(U, 1, axis=1)  # U_{i, j-1}
    
    F_R = np.roll(F, -1, axis=2)
    F_L = np.roll(F, 1, axis=2)
    G_U = np.roll(G, -1, axis=1)
    G_D = np.roll(G, 1, axis=1)
    
    # Flujos numéricos LLF en X e Y
    F_num_x1 = 0.5 * (F_R + F) - 0.5 * cx_max * (U_R - U) # Interfaz i+1/2
    F_num_x2 = 0.5 * (F + F_L) - 0.5 * cx_max * (U - U_L) # Interfaz i-1/2
    
    G_num_y1 = 0.5 * (G_U + G) - 0.5 * cy_max * (U_U - U) # Interfaz j+1/2
    G_num_y2 = 0.5 * (G + G_D) - 0.5 * cy_max * (U - U_D) # Interfaz j-1/2
    
    # Actualización conservativa
    dU = -(dt / dx) * (F_num_x1 - F_num_x2) - (dt / dy) * (G_num_y1 - G_num_y2)
    return U + dU, np.max([cx_max, cy_max])

# ==========================================
# 4. Inicialización (Rotor de Balsara)
# ==========================================
U = np.zeros((8, Ny, Nx))
r = np.sqrt(X**2 + Y**2)
r0, r1, v0 = 0.1, 0.115, 2.0

for i in range(Ny):
    for j in range(Nx):
        rad = r[i, j]
        Bx_init = 5.0 / np.sqrt(4.0 * np.pi)
        
        if rad <= r0:
            rho, vx, vy = 10.0, -v0 * (Y[i, j] / r0), v0 * (X[i, j] / r0)
        elif rad > r0 and rad < r1:
            f = (r1 - rad) / (r1 - r0)
            rho, vx, vy = 1.0 + 9.0 * f, -f * v0 * (Y[i, j] / rad), f * v0 * (X[i, j] / rad)
        else:
            rho, vx, vy = 1.0, 0.0, 0.0
            
        U[0, i, j] = rho
        U[1, i, j] = rho * vx
        U[2, i, j] = rho * vy
        U[3, i, j] = 0.0  # rho * vz
        U[5, i, j] = Bx_init
        U[6, i, j] = 0.0  # By
        U[7, i, j] = 0.0  # Bz
        
        # Energía Total: E = p/(gamma-1) + Ek + Eb
        p_init = 1.0
        Ek = 0.5 * rho * (vx**2 + vy**2)
        Eb = 0.5 * (Bx_init**2)
        U[4, i, j] = (p_init / (gamma - 1.0)) + Ek + Eb

# ==========================================
# 5. Integración Temporal (Método RK2 TVD)
# ==========================================
t = 0.0
step = 0
print("Iniciando simulación del Rotor Magnético con TVD RK2...")
start_time = time.time()

while t < t_end:
    # 1. Calcular variables primitivas del estado actual para hallar el dt
    rho, vx, vy, vz, p, Bx, By, Bz = get_primitives(U)
    cf_x, cf_y = compute_fast_magnetosonic(rho, vx, vy, vz, p, Bx, By, Bz)
    v_max = np.max(np.sqrt(vx**2 + vy**2) + np.sqrt(cf_x**2 + cf_y**2))
    
    # 2. Paso de tiempo dinámico (CFL)
    dt = CFL * dx / v_max
    
    # Ajustar dt final para llegar exacto a t_end
    if t + dt > t_end:
        dt = t_end - t
        
    # 3. Esquema TVD RK2
    # --- Paso 1: Predictor ---
    U_star, _ = rusanov_update(U, dx, dy, dt)
    
    # --- Paso 2: Corrector ---
    # Nota: Usamos el mismo dt del estado original para mantener consistencia temporal
    U_star2, _ = rusanov_update(U_star, dx, dy, dt)
    
    # --- Paso 3: Promedio de estados ---
    U = 0.5 * U + 0.5 * U_star2
    
    t += dt
    step += 1
    
    if step % 20 == 0:
        print(f"Paso: {step}, Tiempo: {t:.4f} / {t_end:.4f}")

print(f"Simulación completada en {time.time() - start_time:.2f} segundos.")
# ==========================================
# 6. Post-procesamiento y Visualización
# ==========================================
rho_final, _, _, _, p_final, _, _, _ = get_primitives(U)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Densidad
c1 = axes[0].pcolormesh(X, Y, rho_final, cmap='viridis', shading='auto')
axes[0].set_title(r'Densidad ($\rho$) en $t = 0.15$')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
fig.colorbar(c1, ax=axes[0])

# Plot Presión
c2 = axes[1].pcolormesh(X, Y, p_final, cmap='plasma', shading='auto')
axes[1].set_title(r'Presión ($P$) en $t = 0.15$')
axes[1].set_xlabel('X')
fig.colorbar(c2, ax=axes[1])

plt.tight_layout()
plt.show()