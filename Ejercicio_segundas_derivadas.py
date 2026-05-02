import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Primera parte: probar la derivada segunda centrada
# ------------------------------------------------------------

def derivada2_centrada(funcion, punto, paso):
    """Aproximación de f''(x) por diferencias finitas centradas O(h^2)"""
    return (funcion(punto + paso) - 2.0*funcion(punto) + funcion(punto - paso)) / (paso**2)

# Función de prueba
def func_seno(x):
    return np.sin(x)

x_prueba = 1.0
h_prueba = 1e-3

aprox = derivada2_centrada(func_seno, x_prueba, h_prueba)
print(f"f''({x_prueba}) ≈ {aprox:.10f} (usando h={h_prueba})")
print()

# ------------------------------------------------------------
# Segunda parte: estudio del error vs tamaño de paso
# ------------------------------------------------------------

# Segunda derivada exacta
def f_seno_segunda(x):
    return -np.sin(x)

# Parámetros
punto_fijo = 1.0

# Generamos tamaños de paso (logarítmicamente espaciados)
pasos = np.logspace(-8, -1, 200)   # 200 puntos entre 1e-8 y 1e-1

# Calculamos el error absoluto para cada paso
valores_numericos = derivada2_centrada(func_seno, punto_fijo, pasos)
valor_teorico = f_seno_segunda(punto_fijo)
errores = np.abs(valores_numericos - valor_teorico)

# Encontrar el paso que minimiza el error (mínimo numérico)
idx_min = np.argmin(errores)
h_min = pasos[idx_min]
error_min = errores[idx_min]

# Para el ajuste lineal, tomamos los datos después del mínimo
# (zona donde domina el error de truncamiento)
errores_despues = errores[idx_min:]
pasos_despues = pasos[idx_min:]

# Tomamos un subconjunto para evitar el ruido inicial (saltarnos los primeros puntos)
inicio_ajuste = min(5, len(pasos_despues) - 2)
pasos_ajuste = pasos_despues[inicio_ajuste:]
errores_ajuste = errores_despues[inicio_ajuste:]

# Ajuste lineal en escala logarítmica: log10(error) = m * log10(h) + b
coefs = np.polyfit(np.log10(pasos_ajuste), np.log10(errores_ajuste), 1)
pendiente = coefs[0]
intercepto = coefs[1]

# Generamos la recta del ajuste para graficar
recta_ajuste = 10**intercepto * (pasos_ajuste ** pendiente)

# ------------------------------------------------------------
# Gráfica
# ------------------------------------------------------------
plt.figure(figsize=(9, 6))

plt.loglog(pasos, errores, 'b-', linewidth=2, label='Error absoluto medido')
plt.loglog(pasos_ajuste, recta_ajuste, 'r--', linewidth=2,
           label=f'Ajuste: pendiente = {pendiente:.3f}')

plt.xlabel('Tamaño de paso $h$', fontsize=12)
plt.ylabel('Error absoluto $|f_{num}'' - f_{exacta}''|$', fontsize=12)
plt.title('Comportamiento del error en la segunda derivada (función seno)', fontsize=13)
plt.grid(True, which='both', linestyle=':', alpha=0.7)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Resultados impresos
# ------------------------------------------------------------
print("=== Análisis del orden de convergencia ===")
print(f"Pendiente obtenida en el ajuste: {pendiente:.4f}")
print("Pendiente teórica esperada: 2.0000")
print(f"¿Es cercana a 2? {'✓ Sí (diferencia < 0.2)' if abs(pendiente - 2.0) < 0.2 else '✗ No'}")
print()
print(f"Paso óptimo (error mínimo): h ≈ {h_min:.2e}  con error ≈ {error_min:.2e}")