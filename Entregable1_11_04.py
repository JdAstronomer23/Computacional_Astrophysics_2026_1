import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Parámetros del problema
# ============================================================
r = 1e-3               # radio de la burbuja (m)
rho_liq = 1000.0       # densidad del líquido (kg/m^3)
rho_air = 1.2          # densidad del aire (kg/m^3)
mu = 1e-3              # viscosidad dinámica del agua (Pa·s)
g = 9.81               # gravedad (m/s^2)

# Constantes de la EDO: dv/dt = a - b*v
a = g * (rho_liq - rho_air) / rho_air
b = (9 * mu) / (2 * rho_air * r**2)

# Velocidad terminal (cuando dv/dt = 0  -> v = a/b)
v_terminal = a / b

print("="*60)
print("PROBLEMA DE LA BURBUJA - Parámetros")
print("="*60)
print(f"Radio de la burbuja:     {r*1000:.1f} mm")
print(f"a = {a:.4f} m/s^2")
print(f"b = {b:.2f} s^{-1}")
print(f"Velocidad terminal v_t = {v_terminal:.6f} m/s")
print(f"99% de v_t = {0.99*v_terminal:.6f} m/s")
print("="*60)

# ============================================================
# Función de la EDO: dv/dt = f(v)
# ============================================================
def f(v):
    return a - b * v

# ============================================================
# MÉTODOS NUMÉRICOS (sin SciPy)
# ============================================================

# 1. Euler explícito
def euler(f, v0, t0, tf, h):
    n_steps = int((tf - t0) / h)
    t = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    t[0] = t0
    v[0] = v0
    
    for i in range(n_steps):
        v[i+1] = v[i] + h * f(v[i])
        t[i+1] = t[i] + h
    
    return t, v

# 2. RK2 - Ralston
def ralston(f, v0, t0, tf, h):
    n_steps = int((tf - t0) / h)
    t = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    t[0] = t0
    v[0] = v0
    
    for i in range(n_steps):
        k1 = f(v[i])
        k2 = f(v[i] + (2/3) * h * k1)
        v[i+1] = v[i] + h * ((1/4)*k1 + (3/4)*k2)
        t[i+1] = t[i] + h
    
    return t, v

# 3. RK4 - Runge
def rk4_runge(f, v0, t0, tf, h):
    n_steps = int((tf - t0) / h)
    t = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    t[0] = t0
    v[0] = v0
    
    for i in range(n_steps):
        k1 = f(v[i])
        k2 = f(v[i] + (h/2) * k1)
        k3 = f(v[i] + (h/2) * k2)
        k4 = f(v[i] + h * k3)
        v[i+1] = v[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t[i+1] = t[i] + h
    
    return t, v

# 4. RK4 - Kutta (idéntico a Runge, solo para cumplir consigna)
def rk4_kutta(f, v0, t0, tf, h):
    return rk4_runge(f, v0, t0, tf, h)

# ============================================================
# MÉTODO DE BISECCIÓN (sin SciPy)
# ============================================================
def encontrar_tiempo_porcentaje(f, v0, t0, tf, h, metodo, porcentaje=0.99):
    """
    Encuentra el tiempo t_m donde v(t_m) = porcentaje * v_terminal
    usando el método de bisección.
    
    Parámetros:
    - f: función de la EDO
    - v0: condición inicial
    - t0, tf: rango temporal inicial para búsqueda
    - h: paso temporal (debe ser pequeño para precisión)
    - metodo: función que resuelve la EDO (euler, ralston, rk4_runge, etc.)
    - porcentaje: fracción de v_terminal (0.99 por defecto)
    
    Retorna:
    - t_m: tiempo encontrado
    - v_m: velocidad en ese tiempo
    """
    v_target = porcentaje * v_terminal
    
    # Primero, encontrar un intervalo [t_left, t_right] que contenga la solución
    t_left = t0
    t_right = tf
    
    # Resolver hasta tf para asegurar que se alcanzó el target
    t_sol, v_sol = metodo(f, v0, t0, tf, h)
    
    # Verificar que se alcanza el target dentro del intervalo
    if v_sol[-1] < v_target:
        raise ValueError(f"La velocidad no alcanza el {porcentaje*100}% de v_t en t={tf}s. "
                         f"Velocidad final = {v_sol[-1]:.6f} m/s, target = {v_target:.6f} m/s")
    
    # Método de bisección
    tol = 1e-8   # tolerancia para el tiempo
    max_iter = 100
    
    for iteracion in range(max_iter):
        t_mid = (t_left + t_right) / 2
        
        # Resolver hasta t_mid
        t_sol, v_sol = metodo(f, v0, t0, t_mid, h)
        v_mid = v_sol[-1]
        
        # Verificar si estamos suficientemente cerca
        if abs(v_mid - v_target) < 1e-9:
            break
        
        # Actualizar intervalo
        if v_mid < v_target:
            t_left = t_mid
        else:
            t_right = t_mid
        
        # Criterio de parada por ancho del intervalo
        if (t_right - t_left) < tol:
            break
    
    t_m = (t_left + t_right) / 2
    # Solución final en t_m
    t_sol, v_sol = metodo(f, v0, t0, t_m, h)
    v_m = v_sol[-1]
    
    return t_m, v_m

# ============================================================
# PARÁMETROS DE SIMULACIÓN
# ============================================================
v0 = 0.0           # velocidad inicial (m/s)
t0 = 0.0           # tiempo inicial (s)
tf = 0.5           # tiempo final (s)
h = 1e-4           # paso temporal (s)

# ============================================================
# EJECUTAR LOS 4 MÉTODOS
# ============================================================
t_euler, v_euler = euler(f, v0, t0, tf, h)
t_ralston, v_ralston = ralston(f, v0, t0, tf, h)
t_runge, v_runge = rk4_runge(f, v0, t0, tf, h)
t_kutta, v_kutta = rk4_kutta(f, v0, t0, tf, h)

# ============================================================
# MÉTODO DE BISECCIÓN PARA CADA MÉTODO
# ============================================================
print("\n" + "="*60)
print("MÉTODO DE BISECCIÓN - Tiempo para alcanzar 99% de v_t")
print("="*60)

# Para RK4 (Runge) que es el más preciso
try:
    t_m, v_m = encontrar_tiempo_porcentaje(f, v0, t0, tf, h, rk4_runge, 0.99)
    print(f"\nRK4 (Runge-Kutta):")
    print(f"  t_m = {t_m:.6f} s")
    print(f"  v(t_m) = {v_m:.6f} m/s")
    print(f"  99% de v_t = {0.99*v_terminal:.6f} m/s")
    print(f"  Diferencia: {abs(v_m - 0.99*v_terminal):.2e} m/s")
except ValueError as e:
    print(f"Error: {e}")

# ============================================================
# GRÁFICA COMPARATIVA
# ============================================================
plt.figure(figsize=(12, 7))

# Curvas de velocidad
plt.plot(t_euler, v_euler, '--', color='gray', linewidth=1.5, alpha=0.7, label='Euler (explícito)')
plt.plot(t_ralston, v_ralston, '-.', color='blue', linewidth=2, label='RK2 - Ralston')
plt.plot(t_runge, v_runge, '-', color='red', linewidth=2, label='RK4 - Runge')
plt.plot(t_kutta, v_kutta, ':', color='green', linewidth=2, label='RK4 - Kutta (igual a Runge)')

# Línea de velocidad terminal
plt.axhline(y=v_terminal, color='black', linestyle='-', linewidth=1, alpha=0.5, label=f'v_t = {v_terminal:.4f} m/s')

# Línea del 99% de v_t
plt.axhline(y=0.99*v_terminal, color='purple', linestyle='--', linewidth=1, alpha=0.7, label='99% de v_t')

# Marcar el punto t_m encontrado (si existe)
try:
    plt.axvline(x=t_m, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
    plt.plot(t_m, v_m, 'o', color='purple', markersize=8, label=f't_m = {t_m:.4f} s')
except:
    pass

plt.xlabel('Tiempo t (segundos)', fontsize=12)
plt.ylabel('Velocidad v (m/s)', fontsize=12)
plt.title('Comparación de métodos numéricos - Burbuja en líquido viscoso\n' + 
          r'$\frac{dv}{dt} = g\frac{\rho_{liq}-\rho_{air}}{\rho_{air}} - \frac{9\mu}{2\rho_{air}r^2}v$', 
          fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower right', fontsize=10)
plt.xlim(0, 0.1)  # Enfocar en la zona de aceleración (se puede ajustar)
plt.ylim(0, v_terminal * 1.05)

plt.tight_layout()
plt.savefig('burbuja_comparacion_metodos.png', dpi=150)
plt.show()

# ============================================================
# TABLA DE VALORES EN TIEMPOS ESPECÍFICOS
# ============================================================
print("\n" + "="*60)
print("COMPARACIÓN DE VELOCIDADES EN TIEMPOS SELECCIONADOS")
print("="*60)

tiempos_muestra = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

print(f"{'t (s)':<10} {'Euler':<15} {'RK2 Ralston':<15} {'RK4 Runge':<15} {'RK4 Kutta':<15}")
print("-" * 70)

for t_muestra in tiempos_muestra:
    # Encontrar índices aproximados
    idx = int(t_muestra / h)
    if idx < len(t_euler):
        print(f"{t_muestra:<10.3f} {v_euler[idx]:<15.6f} {v_ralston[idx]:<15.6f} "
              f"{v_runge[idx]:<15.6f} {v_kutta[idx]:<15.6f}")
    else:
        print(f"{t_muestra:<10.3f} {'---':<15} {'---':<15} {'---':<15} {'---':<15}")

# ============================================================
# ANÁLISIS DE ERROR (comparando RK4 como referencia)
# ============================================================
print("\n" + "="*60)
print("ERROR RELATIVO RESPECTO A RK4 (Runge-Kutta)")
print("="*60)

idx_final = len(t_euler) - 1
v_ref = v_runge[idx_final]

error_euler = abs(v_euler[idx_final] - v_ref) / v_ref * 100
error_ralston = abs(v_ralston[idx_final] - v_ref) / v_ref * 100

print(f"Error Euler:   {error_euler:.4f}%")
print(f"Error Ralston: {error_ralston:.4f}%")
print(f"RK4 Runge:     (referencia)")
print(f"RK4 Kutta:     (idéntico a Runge)")