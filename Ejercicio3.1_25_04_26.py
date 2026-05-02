import numpy as np

def f(l, theta):
    return (18 - 2*l + 2*l*np.cos(theta)) * l * np.sin(theta)

# Prueba manual de la función en el punto máximo conocido
print("Prueba f(9, pi/4):", f(9, np.pi/4))   # Debe dar ~81.0

def golden_section_max(func, a, b, tol=1e-3, max_iter=100):
    phi = (np.sqrt(5) - 1) / 2
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    f1 = func(x1)
    f2 = func(x2)
    it = 0
    print(f"Inicio: a={a:.4f}, b={b:.4f}, x1={x1:.4f}, x2={x2:.4f}, f1={f1:.4f}, f2={f2:.4f}")
    while abs(b - a) > tol and it < max_iter:
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
        it += 1
        print(f"Iter {it}: a={a:.4f}, b={b:.4f}, x1={x1:.4f}, x2={x2:.4f}, f1={f1:.4f}, f2={f2:.4f}")
    return (a + b) / 2, it

# Optimizar l con theta fijo = pi/6
theta_fijo = np.pi/6
g_l = lambda l: f(l, theta_fijo)
l_opt, it_l = golden_section_max(g_l, 0, 9, tol=0.05)
print(f"\nResultado parcial: l_opt = {l_opt:.6f}, f(l_opt, theta_fijo) = {f(l_opt, theta_fijo):.6f}, iter={it_l}\n")

# Optimizar theta con l fijo
g_theta = lambda theta: f(l_opt, theta)
theta_opt, it_theta = golden_section_max(g_theta, 0, np.pi/2, tol=0.05)
print(f"\nResultado final: l* = {l_opt:.6f}, theta* = {theta_opt:.6f} rad ({np.degrees(theta_opt):.2f}°), A* = {f(l_opt, theta_opt):.6f}")
print(f"Iteraciones l: {it_l}, θ: {it_theta}")