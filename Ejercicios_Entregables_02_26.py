# ============================================
# Ejercicio 1: Aceleración orbital (diferencias finitas)
# ============================================

delta_t = 0.5                     # intervalo de tiempo (s)
tiempos = [0.0, 0.5, 1.0, 1.5, 2.0]   # s
velocidades = [200.0, 205.2, 210.8, 216.9, 223.5]  # km/s

# a) Diferencia hacia adelante en t0
acel_inicio = (velocidades[1] - velocidades[0]) / delta_t

# b) Diferencia hacia atrás en t4
acel_final = (velocidades[4] - velocidades[3]) / delta_t

# c) Diferencia centrada en t2
acel_centro = (velocidades[3] - velocidades[1]) / (2 * delta_t)

print("--- Ejercicio 1 ---")
print(f"a(t0 = {tiempos[0]} s) = {acel_inicio:.2f} km/s²")
print(f"a(t4 = {tiempos[4]} s) = {acel_final:.2f} km/s²")
print(f"a(t2 = {tiempos[2]} s) = {acel_centro:.2f} km/s²\n")

# ============================================
# Ejercicio 2: Gradiente de presión en una estrella
# ============================================

radios = [1.0, 1.1, 1.2, 1.3, 1.4]       # en 10^3 km
presiones = [10.00, 9.15, 8.20, 7.12, 5.90]  # unidades arbitrarias
step_r = 0.1                                # en 10^3 km

# Punto más externo i=4 (índice 4): usar P4, P3, P2
P_4 = presiones[4]
P_3 = presiones[3]
P_2 = presiones[2]

grad_p_extremo = (3*P_4 - 4*P_3 + P_2) / (2 * step_r)

print("--- Ejercicio 2 ---")
print(f"dP/dr en r = {radios[4]} (10^3 km) = {grad_p_extremo:.4f} (unid. presión)/(10^3 km)\n")

# ============================================
# Ejercicio 3: Gradientes de corrimiento al rojo (redshift)
# ============================================

distancias = [10, 12, 14, 16, 18]        # Mpc
redshifts = [0.0020, 0.0035, 0.0055, 0.0080, 0.0110]
h_D = 2                                  # paso en Mpc (espaciado constante)

# Punto medio D=14 (índice 2): usar vecinos i-1, i, i+1
z_izq = redshifts[1]   # z(12)
z_med = redshifts[2]   # z(14)
z_der = redshifts[3]   # z(16)

# Primera derivada centrada
deriv1 = (z_der - z_izq) / (2 * h_D)

# Segunda derivada centrada
deriv2 = (z_der - 2*z_med + z_izq) / (h_D**2)

print("--- Ejercicio 3 ---")
print("Cálculo de dz/dD:")
print(f"dz/dD ≈ (z(D+{h_D}) - z(D-{h_D})) / (2*{h_D})")
print(f"     ≈ ({z_der:.4f} - {z_izq:.4f}) / {2*h_D}")
print(f"     ≈ {deriv1:.6f} 1/Mpc\n")

print("Cálculo de d²z/dD²:")
print(f"d²z/dD² ≈ (z(D+{h_D}) - 2z(D) + z(D-{h_D})) / {h_D}²")
print(f"       ≈ ({z_der:.4f} - 2*{z_med:.4f} + {z_izq:.4f}) / {h_D**2}")
print(f"       ≈ {deriv2:.6f} 1/Mpc²\n")

# ============================================
# Ejercicio 5: Análisis de error y convergencia (orden p=2)
# ============================================

orden = 2
paso = 0.1
constante = 1.0

error_h = constante * (paso ** orden)
error_h_mitad = constante * ((paso/2) ** orden)

cociente = error_h_mitad / error_h

print("--- Ejercicio 5 (Pregunta 2) ---")
print(f"E(h)      = {error_h:.6f}")
print(f"E(h/2)    = {error_h_mitad:.6f}")
print(f"E(h/2)/E(h) = {cociente:.6f}")
print(f"Factor de reducción = {1/cociente:.0f}  (el error se reduce 4 veces)")