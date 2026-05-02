import numpy as np
import matplotlib.pyplot as plt

# Función y su segunda derivada exacta
func = np.sin
exact_d2 = lambda x: -np.sin(x)

# Fórmulas de diferencias finitas para la segunda derivada
def forward_diff2(f, x, h):
    return (f(x + 2*h) - 2*f(x + h) + f(x)) / h**2   # Error O(h)

def backward_diff2(f, x, h):
    return (f(x) - 2*f(x - h) + f(x - 2*h)) / h**2   # Error O(h)

def centered_diff2(f, x, h):
    return (f(x + h) - 2*f(x) + f(x - h)) / h**2     # Error O(h^2)

# Parámetros de la prueba
h_list = np.logspace(-3, -1, 12)   # h = 0.001 a 0.1, 12 puntos
x_point = 1.0                      # Punto fijo de evaluación

# Almacenar errores absolutos
errors_fwd = []
errors_bwd = []
errors_ctr = []

for step in h_list:
    valor_exacto = exact_d2(x_point)
    
    error_f = abs(valor_exacto - forward_diff2(func, x_point, step))
    error_b = abs(valor_exacto - backward_diff2(func, x_point, step))
    error_c = abs(valor_exacto - centered_diff2(func, x_point, step))
    
    errors_fwd.append(error_f)
    errors_bwd.append(error_b)
    errors_ctr.append(error_c)

# Convertir a arrays
errors_fwd = np.array(errors_fwd)
errors_bwd = np.array(errors_bwd)
errors_ctr = np.array(errors_ctr)

# Pendientes en escala log-log
pend_fwd = np.polyfit(np.log(h_list), np.log(errors_fwd), 1)[0]
pend_bwd = np.polyfit(np.log(h_list), np.log(errors_bwd), 1)[0]
pend_ctr = np.polyfit(np.log(h_list), np.log(errors_ctr), 1)[0]

print(f"Pendiente Forward  = {pend_fwd:.3f} (teórica ~1)")
print(f"Pendiente Backward = {pend_bwd:.3f} (teórica ~1)")
print(f"Pendiente Centered = {pend_ctr:.3f} (teórica ~2)")

# Gráfica comparativa
plt.figure(figsize=(10, 6))
plt.loglog(h_list, errors_fwd, '--o', label='Diferencias hacia adelante (O(h))')
plt.loglog(h_list, errors_bwd, '--s', label='Diferencias hacia atrás (O(h))')
plt.loglog(h_list, errors_ctr, '--^', label='Diferencias centradas (O(h²))')
plt.xlabel('Tamaño de paso (h)')
plt.ylabel('Error absoluto')
plt.title('Escalamiento del error en la segunda derivada numérica')
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.legend()
plt.show()