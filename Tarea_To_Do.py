# ============================================================================
# 1) Conversión de fecha (mes, día) a día del año (base 0)
# ============================================================================

import numpy as np

def fecha_a_dias(mes, dia):
    mes = np.asarray(mes, dtype=int)
    dia = np.asarray(dia, dtype=int)
    # días por mes para 2017 (no bisiesto)
    dias_por_mes = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=int)
    inicio_mes = np.concatenate(([0], np.cumsum(dias_por_mes)[:-1]))
    return inicio_mes[mes - 1] + (dia - 1)   # 1 de enero → 0

# ============================================================================
# 2) Media móvil simple
# ============================================================================

def media_movil(datos, ventana=30):
    datos = np.asarray(datos, dtype=float)
    kernel = np.ones(ventana, dtype=float) / ventana
    return np.convolve(datos, kernel, mode="same")

# ============================================================================
# 3) Detección de outliers por rango intercuartil (IQR)
# ============================================================================

def mascara_outliers_iqr(serie, factor=1.5):
    q1, q3 = np.percentile(serie, [25, 75])
    iqr = q3 - q1
    limite_inf = q1 - factor * iqr
    limite_sup = q3 + factor * iqr
    return (serie < limite_inf) | (serie > limite_sup)

# ============================================================================
# 4) Carga y preprocesamiento inicial (usando genfromtxt)
# ============================================================================

def cargar_y_preprocesar(ruta="kumpula-weather-2017.csv"):
    datos = np.genfromtxt(ruta, delimiter=",", names=True, dtype=None, encoding="utf-8")

    mes = datos["m"].astype(int)
    dia = datos["d"].astype(int)
    temp = datos["Air_temperature_degC"].astype(float)

    dias = fecha_a_dias(mes, dia).astype(float)

    # limpiar valores no finitos
    mascara_valida = np.isfinite(dias) & np.isfinite(temp)
    dias = dias[mascara_valida]
    temp = temp[mascara_valida]

    # ordenar cronológicamente
    indices_orden = np.argsort(dias)
    dias = dias[indices_orden]
    temp = temp[indices_orden]

    # anomalía = temperatura - media móvil 30 días
    media30 = media_movil(temp, ventana=30)
    anomalia = temp - media30

    # outliers basados en anomalía
    outliers = mascara_outliers_iqr(anomalia, factor=1.5)

    # matriz [temp, anomalia]
    matriz_limpia = np.column_stack([temp, anomalia])   # (N,2)
    return matriz_limpia, dias, outliers

# Ejecutar primera carga
matriz_limpia, dias_crudo, mascara_out = cargar_y_preprocesar("kumpula-weather-2017.csv")

print("matriz_limpia shape:", matriz_limpia.shape)   # (N,2)
print("dias_crudo shape:", dias_crudo.shape)         # (N,)
print("outliers detectados:", int(mascara_out.sum()))

# ============================================================================
# 5) Segunda carga alternativa con pandas (para validación)
# ============================================================================

import pandas as pd

df = pd.read_csv("kumpula-weather-2017.csv")

mes_pd = pd.to_numeric(df["m"], errors="coerce").to_numpy()
dia_pd  = pd.to_numeric(df["d"], errors="coerce").to_numpy()
temp_pd = pd.to_numeric(df["Air temperature (degC)"], errors="coerce").to_numpy()

# Convertir a día juliano (0-index)
dias_por_mes = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=int)
inicio_mes = np.concatenate(([0], np.cumsum(dias_por_mes)[:-1]))
dias_julianos = inicio_mes[mes_pd.astype(int) - 1] + (dia_pd.astype(int) - 1)

# Filtrar valores válidos
mascara_validos = np.isfinite(dias_julianos) & np.isfinite(temp_pd)
dias_julianos = dias_julianos[mascara_validos].astype(float)
temp_pd = temp_pd[mascara_validos].astype(float)

# Media móvil de 30 días
media30_pd = np.convolve(temp_pd, np.ones(30)/30, mode="same")
anomalia_pd = temp_pd - media30_pd

matriz_limpia_pd = np.column_stack([temp_pd, anomalia_pd])

print("dias_julianos shape:", dias_julianos.shape)
print("matriz_limpia_pd shape:", matriz_limpia_pd.shape)

# ============================================================================
# 6) Clases con herencia y decorador vectorize
# ============================================================================

from functools import wraps

def vectorizar(func):
    """Decorador para aplicar una función elemento a elemento a arrays."""
    @wraps(func)
    def wrapper(self, x, *args, **kwargs):
        x = np.asarray(x)
        func_vec = np.vectorize(lambda z: func(self, z, *args, **kwargs))
        return func_vec(x)
    return wrapper

class AnalizadorSeriesTemporales:
    def __init__(self, tiempo, valores):
        self.tiempo = np.asarray(tiempo, dtype=float)
        self.valores = np.asarray(valores, dtype=float)

    def suavizar(self, ventana=7):
        kernel = np.ones(ventana, dtype=float) / ventana
        return np.convolve(self.valores, kernel, mode="same")

    @vectorizar
    def identidad(self, x):
        return x

class AnalizadorClima(AnalizadorSeriesTemporales):
    def descomponer_estacional(self):
        # tendencia: polinomio grado 2
        coefs = np.polyfit(self.tiempo, self.valores, deg=2)
        tendencia = np.polyval(coefs, self.tiempo)

        # serie sin tendencia
        sin_tendencia = self.valores - tendencia
        n = len(sin_tendencia)
        f_hat = np.fft.rfft(sin_tendencia)
        amplitudes = np.abs(f_hat)
        if len(amplitudes) > 0:
            amplitudes[0] = 0.0  # eliminar componente DC

        # elegir las k frecuencias más energéticas
        k = min(4, len(f_hat))
        indices_top = np.argsort(amplitudes)[-k:]

        mascara = np.zeros_like(f_hat, dtype=complex)
        mascara[indices_top] = f_hat[indices_top]
        estacional = np.fft.irfft(mascara, n=n)

        residual = self.valores - tendencia - estacional
        return tendencia, estacional, residual

    def pronosticar(self, dias_futuros=30):
        # usar últimos 30 puntos (o menos si hay pocos)
        m = min(30, len(self.tiempo))
        t_ultimos = self.tiempo[-m:]
        y_ultimos = self.valores[-m:]

        # ajuste cúbico local
        coefs_cub = np.polyfit(t_ultimos, y_ultimos, deg=3)
        t_futuro = np.arange(self.tiempo[-1] + 1, self.tiempo[-1] + dias_futuros + 1)

        tendencia_poly = np.polyval(coefs_cub, t_futuro)

        # ruido con desviación típica de los residuos locales
        residuos_local = y_ultimos - np.polyval(coefs_cub, t_ultimos)
        sigma_ruido = np.std(residuos_local)
        np.random.seed(1997)
        ruido = np.random.normal(0.0, sigma_ruido, size=dias_futuros)

        pronostico = tendencia_poly + ruido
        return t_futuro, pronostico

# Instanciar analizador con los días y anomalías (columna 1 de matriz_limpia)
analizador = AnalizadorClima(dias_crudo, matriz_limpia[:, 1])

suavizado7 = analizador.suavizar(ventana=7)
tendencia, estacional, residual = analizador.descomponer_estacional()
futuro_dias, futuro_valores = analizador.pronosticar(dias_futuros=30)

print("suavizado7 shape:", suavizado7.shape)
print("tendencia shape:", tendencia.shape)
print("estacional shape:", estacional.shape)
print("residual shape:", residual.shape)
print("pronóstico shapes:", futuro_dias.shape, futuro_valores.shape)

# Validaciones
assert matriz_limpia.shape[1] == 2
assert len(dias_crudo) == len(matriz_limpia)
assert tendencia.shape == estacional.shape == residual.shape == dias_crudo.shape
assert futuro_dias.shape == futuro_valores.shape == (30,)
print("Sin errores.")

# ============================================================================
# 7) Guardado de archivos procesados
# ============================================================================

# Si por alguna razón no existen las variables, recargar con fallback
if "dias_crudo" not in globals() or "matriz_limpia" not in globals():
    df_fallback = pd.read_csv("kumpula-weather-2017.csv")
    mes_fb = pd.to_numeric(df_fallback["m"], errors="coerce").to_numpy()
    dia_fb = pd.to_numeric(df_fallback["d"], errors="coerce").to_numpy()
    temp_fb = pd.to_numeric(df_fallback["Air temperature (degC)"], errors="coerce").to_numpy()

    dias_por_mes = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    inicio_mes = np.concatenate(([0], np.cumsum(dias_por_mes)[:-1]))
    dias_fb = inicio_mes[mes_fb.astype(int)-1] + (dia_fb.astype(int)-1)

    validos = np.isfinite(dias_fb) & np.isfinite(temp_fb)
    dias_fb = dias_fb[validos].astype(float)
    temp_fb = temp_fb[validos].astype(float)

    media30_fb = np.convolve(temp_fb, np.ones(30)/30, mode="same")
    anomalias_fb = temp_fb - media30_fb
    matriz_limpia = np.column_stack([temp_fb, anomalias_fb])
    dias_crudo = dias_fb

# Variables principales (para claridad)
dias_julianos_clean = np.asarray(dias_crudo, dtype=float)
temperatura = np.asarray(matriz_limpia[:, 0], dtype=float)
anomalia = np.asarray(matriz_limpia[:, 1], dtype=float)

# Suavizado con ventana 7 (para exportar)
temp_suavizada = np.convolve(temperatura, np.ones(7)/7, mode="same")

# Guardar en .npz
np.savez("processed_weather.npz", temps_clean=matriz_limpia, anomalies=anomalia, days=dias_julianos_clean)
print("Guardado: processed_weather.npz")

# Guardar subset de 100 primeras filas en CSV
n_primeros = min(100, len(dias_julianos_clean))
subset = np.column_stack([dias_julianos_clean[:n_primeros],
                          temp_suavizada[:n_primeros],
                          anomalia[:n_primeros]])

np.savetxt(
    "subset.csv",
    subset,
    delimiter=",",
    fmt="%.2f",
    header="days,temp_smooth,anomaly",
    comments=""
)
print("Guardado: subset.csv")

# ============================================================================
# 8) Función robusta de carga y validación
# ============================================================================

def cargar_y_validar(archivo):
    resultado = {"archivo": archivo, "valido": False, "formato": None, "datos": {}}

    if archivo.endswith(".npz"):
        resultado["formato"] = "npz"
        contenido = np.load(archivo)

        for clave in contenido.files:
            arr = np.asarray(contenido[clave])
            resultado["datos"][clave] = arr

        # Validaciones requeridas
        if "temps_clean" not in resultado["datos"]:
            raise ValueError("NPZ inválido: falta la variable 'temps_clean'.")

        tc = resultado["datos"]["temps_clean"]
        if tc.ndim != 2 or tc.shape[1] != 2:
            raise ValueError(f"'temps_clean' con forma {tc.shape}, se esperaba (N,2).")

        for clave, arr in resultado["datos"].items():
            if np.issubdtype(arr.dtype, np.number):
                if np.isnan(arr).any() or np.isinf(arr).any():
                    raise ValueError(f"El arreglo '{clave}' tiene NaN o Inf.")

        resultado["valido"] = True
        return resultado

    elif archivo.endswith(".csv"):
        resultado["formato"] = "csv"
        arr_csv = np.loadtxt(archivo, delimiter=",", skiprows=1)
        if arr_csv.ndim == 1:
            arr_csv = arr_csv.reshape(1, -1)

        if arr_csv.shape[1] != 3:
            raise ValueError(f"CSV inválido: se esperaban 3 columnas, se obtuvo {arr_csv.shape[1]}.")

        if np.isnan(arr_csv).any() or np.isinf(arr_csv).any():
            raise ValueError("El CSV contiene NaN o Inf.")

        resultado["datos"]["tabla"] = arr_csv
        resultado["valido"] = True
        return resultado

    else:
        raise ValueError("Formato no soportado. Use .npz o .csv")

# Probar carga de ambos archivos
info_npz = cargar_y_validar("processed_weather.npz")
info_csv = cargar_y_validar("subset.csv")

print("NPZ válido:", info_npz["valido"], "| claves:", list(info_npz["datos"].keys()))
print("CSV válido:", info_csv["valido"], "| forma:", info_csv["datos"]["tabla"].shape)