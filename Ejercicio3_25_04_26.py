import numpy as np

# Función a maximizar
def f(l, theta):
    return (18 - 2*l + 2*l*np.cos(theta)) * l * np.sin(theta)

# Método de sección áurea (maximización)
def golden_section_max(func, a, b, tol=1e-3, max_iter=100):
    phi = (np.sqrt(5) - 1) / 2
    
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    
    f1 = func(x1)
    f2 = func(x2)
    
    iterations = 0
    
    while abs(b - a) > tol and iterations < max_iter:
        if f1 < f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + phi * (b - a)
            f2 = func(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - phi * (b - a)
            f1 = func(x1)
        
        iterations += 1
    
    return (a + b) / 2, iterations

# Ciclo de coordenadas
def coordinate_descent(tol=0.05, max_iter=50):
    l = 0.0
    theta = np.pi / 6
    
    total_iter = 0
    
    for i in range(max_iter):
        l_old, theta_old = l, theta
        
        # Optimizar l con theta fijo
        g_l = lambda l_var: f(l_var, theta)
        l, it_l = golden_section_max(g_l, 0, 9)
        
        # Optimizar theta con l fijo
        g_theta = lambda th: f(l, th)
        theta, it_theta = golden_section_max(g_theta, 0, np.pi/2)
        
        total_iter += it_l + it_theta
        
        # criterio de convergencia
        if abs(l - l_old) < tol and abs(theta - theta_old) < tol:
            break
    
    return l, theta, f(l, theta), total_iter


# Ejecutar
l_opt, theta_opt, area_max, iterations = coordinate_descent()

print("l óptimo:", l_opt)
print("theta óptimo (rad):", theta_opt)
print("theta óptimo (deg):", np.degrees(theta_opt))
print("Área máxima:", area_max)
print("Iteraciones totales:", iterations)