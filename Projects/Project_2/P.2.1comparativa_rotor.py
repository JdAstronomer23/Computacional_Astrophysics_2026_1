import numpy as np
import matplotlib.pyplot as plt
import pyPLUTO as pp
import os
import time

Nx, Ny = 200, 200  
x = np.linspace(-0.5, 0.5, Nx)
y = np.linspace(-0.5, 0.5, Ny)
X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]
gamma = 1.4
CFL = 0.4
t_end = 0.15

def get_primitives(U):
    rho = U[0]
    vx, vy, vz = U[1]/rho, U[2]/rho, U[3]/rho
    Bx, By, Bz = U[5], U[6], U[7]
    E_k = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    E_b = 0.5 * (Bx**2 + By**2 + Bz**2)
    p = (gamma - 1.0) * (U[4] - E_k - E_b)
    p = np.maximum(p, 1e-9)
    return rho, vx, vy, vz, p, Bx, By, Bz

def compute_fast_magnetosonic(rho, vx, vy, vz, p, Bx, By, Bz):
    a2 = gamma * p / rho
    b2 = (Bx**2 + By**2 + Bz**2) / rho
    cf_x = np.sqrt(0.5 * (a2 + b2 + np.sqrt((a2 + b2)**2 - 4 * a2 * (Bx**2/rho))))
    cf_y = np.sqrt(0.5 * (a2 + b2 + np.sqrt((a2 + b2)**2 - 4 * a2 * (By**2/rho))))
    return cf_x, cf_y

def compute_fluxes(U):
    rho, vx, vy, vz, p, Bx, By, Bz = get_primitives(U)
    p_star = p + 0.5 * (Bx**2 + By**2 + Bz**2)
    v_dot_B = vx*Bx + vy*By + vz*Bz
    
    F = np.zeros_like(U)
    F[0], F[1], F[2], F[3] = rho*vx, rho*vx**2 + p_star - Bx**2, rho*vx*vy - Bx*By, rho*vx*vz - Bx*Bz
    F[4], F[6], F[7] = (U[4] + p_star)*vx - Bx*v_dot_B, vx*By - vy*Bx, vx*Bz - vz*Bx
    
    G = np.zeros_like(U)
    G[0], G[1], G[2], G[3] = rho*vy, rho*vy*vx - By*Bx, rho*vy**2 + p_star - By**2, rho*vy*vz - By*Bz
    G[4], G[5], G[7] = (U[4] + p_star)*vy - By*v_dot_B, vy*Bx - vx*By, vy*Bz - vz*By
    return F, G, rho, vx, vy, vz, p, Bx, By, Bz

def rusanov_update(U, dx, dy, dt):
    F, G, rho, vx, vy, vz, p, Bx, By, Bz = compute_fluxes(U)
    cf_x, cf_y = compute_fast_magnetosonic(rho, vx, vy, vz, p, Bx, By, Bz)
    cx_max, cy_max = np.max(np.abs(vx) + cf_x), np.max(np.abs(vy) + cf_y)
    
    U_R, U_L = np.roll(U, -1, axis=2), np.roll(U, 1, axis=2)
    U_U, U_D = np.roll(U, -1, axis=1), np.roll(U, 1, axis=1)
    
    F_num_x1 = 0.5 * (np.roll(F, -1, axis=2) + F) - 0.5 * cx_max * (U_R - U)
    F_num_x2 = 0.5 * (F + np.roll(F, 1, axis=2)) - 0.5 * cx_max * (U - U_L)
    G_num_y1 = 0.5 * (np.roll(G, -1, axis=1) + G) - 0.5 * cy_max * (U_U - U)
    G_num_y2 = 0.5 * (G + np.roll(G, 1, axis=1)) - 0.5 * cy_max * (U - U_D)
    
    dU = -(dt / dx) * (F_num_x1 - F_num_x2) - (dt / dy) * (G_num_y1 - G_num_y2)
    return U + dU

U = np.zeros((8, Ny, Nx))
r = np.sqrt(X**2 + Y**2)
r0, r1, v0 = 0.1, 0.115, 2.0
Bx_init = 5.0 / np.sqrt(4.0 * np.pi)

for i in range(Ny):
    for j in range(Nx):
        rad = r[i, j]
        if rad <= r0:
            rho, vx, vy = 10.0, -v0 * (Y[i, j] / r0), v0 * (X[i, j] / r0)
        elif rad > r0 and rad < r1:
            f = (r1 - rad) / (r1 - r0)
            rho, vx, vy = 1.0 + 9.0 * f, -f * v0 * (Y[i, j] / rad), f * v0 * (X[i, j] / rad)
        else:
            rho, vx, vy = 1.0, 0.0, 0.0
            
        U[0, i, j], U[1, i, j], U[2, i, j], U[5, i, j] = rho, rho*vx, rho*vy, Bx_init
        U[4, i, j] = (1.0 / (gamma - 1.0)) + 0.5*rho*(vx**2 + vy**2) + 0.5*(Bx_init**2)


print("Corriendo simulación en Python (TVD RK2)...")
t = 0.0
while t < t_end:
    rho, vx, vy, vz, p, Bx, By, Bz = get_primitives(U)
    cf_x, cf_y = compute_fast_magnetosonic(rho, vx, vy, vz, p, Bx, By, Bz)
    v_max = np.max(np.sqrt(vx**2 + vy**2) + np.sqrt(cf_x**2 + cf_y**2))
    dt = CFL * dx / v_max
    if t + dt > t_end: dt = t_end - t
    
    # TVD RK2
    U_star = rusanov_update(U, dx, dy, dt)
    U_star2 = rusanov_update(U_star, dx, dy, dt)
    U = 0.5 * U + 0.5 * U_star2
    t += dt

rho_python, _, _, _, _, _, _, _ = get_primitives(U)
print("Simulación en Python finalizada.")


print("Cargando archivo final de PLUTO (data.0010.dbl)...")
try:
    # Agregamos una barra diagonal extra '/' al final de la ruta para que pyPLUTO no concatene mal los strings
    ruta_actual = os.getcwd() + '/'
    data_pluto = pp.pload.pload(10, w_dir=ruta_actual)
    rho_pluto = data_pluto.rho.T
except Exception as e:
    print(f"\n[ERROR] No se pudo cargar pyPLUTO: {e}")
    print("Verifica si el archivo dbl.out y grid.out existen en esta misma carpeta.")
    exit()


diff_rho = rho_python - rho_pluto
dA = dx * dy
L1_rho = np.sum(np.abs(diff_rho)) * dA
L2_rho = np.sqrt(np.sum(diff_rho**2) * dA)

print(f"\n--- RESULTADOS FORMALES ---")
print(f"Norma L1 del Error: {L1_rho:.6f}")
print(f"Norma L2 del Error: {L2_rho:.6f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
c1 = axes[0].pcolormesh(x, y, rho_pluto, cmap='viridis', shading='auto')
axes[0].set_title(r'PLUTO - Densidad ($\rho$)')
axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
fig.colorbar(c1, ax=axes[0])

c2 = axes[1].pcolormesh(x, y, rho_python, cmap='viridis', shading='auto')
axes[1].set_title(r'Python - Densidad ($\rho$)')
axes[1].set_xlabel('X')
fig.colorbar(c2, ax=axes[1])

c3 = axes[2].pcolormesh(x, y, np.abs(diff_rho), cmap='inferno', shading='auto')
axes[2].set_title('Diferencia Absoluta |Py - PLUTO|')
axes[2].set_xlabel('X')
fig.colorbar(c3, ax=axes[2])

plt.tight_layout()
plt.savefig('comparativa_densidad_rotor.png', dpi=300)
plt.show()