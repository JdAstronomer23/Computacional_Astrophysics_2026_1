import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Función a derivar y su derivada analítica
# ------------------------------------------------------------
def mi_funcion(x):
    return np.exp(x)

def derivada_exacta(x):
    return np.exp(x)

# ------------------------------------------------------------
# 2) Esquemas de diferencias finitas para primera derivada
# ------------------------------------------------------------
def adelante(f, x, h):
    return (f(x + h) - f(x)) / h

def atras(f, x, h):
    return (f(x) - f(x - h)) / h

def centrada(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)

# ------------------------------------------------------------
# 3) Parámetros de la prueba
# ------------------------------------------------------------
punto_evaluacion = 1.0
h_min = 1e-15
h_max = 1e-1
num_pasos = 100000   # mismo número que el original

# Generamos espaciado lineal (mismo método)
h_vals = np.linspace(h_min, h_max, num_pasos)

# ------------------------------------------------------------
# 4) Cálculo de errores absolutos
# ------------------------------------------------------------
valor_real = derivada_exacta(punto_evaluacion)

err_adelante = np.abs(valor_real - adelante(mi_funcion, punto_evaluacion, h_vals))
err_atras    = np.abs(valor_real - atras(mi_funcion, punto_evaluacion, h_vals))
err_centrada = np.abs(valor_real - centrada(mi_funcion, punto_evaluacion, h_vals))

# ------------------------------------------------------------
# 5) Gráfica en escala log-log (eje x invertido)
# ------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.loglog(h_vals, err_atras,    'b-',  linewidth=1.5, label='Diferencias hacia atrás')
plt.loglog(h_vals, err_adelante, 'r--', linewidth=1.2, label='Diferencias hacia adelante')
plt.loglog(h_vals, err_centrada, 'g-.', linewidth=1.8, label='Diferencias centradas')

plt.xlabel(r'Tamaño de paso $h$', fontsize=12)
plt.ylabel('Error absoluto', fontsize=12)
plt.title('Error de derivación numérica para $f(x)=e^x$ en $x=1$', fontsize=13)
plt.xlim(h_max, h_min)   # invertir eje: de grande a pequeño
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6) Encontrar los h óptimos (mínimo error)
# ------------------------------------------------------------
idx_adelante = np.argmin(err_adelante)
idx_atras    = np.argmin(err_atras)
idx_centrada = np.argmin(err_centrada)

print("="*50)
print("Resultados de optimización del paso h:")
print("-"*50)
print(f" Adelante → error mínimo = {err_adelante[idx_adelante]:.3e}   en h = {h_vals[idx_adelante]:.3e}")
print(f" Atrás    → error mínimo = {err_atras[idx_atras]:.3e}   en h = {h_vals[idx_atras]:.3e}")
print(f" Centrada → error mínimo = {err_centrada[idx_centrada]:.3e}   en h = {h_vals[idx_centrada]:.3e}")
print("="*50)