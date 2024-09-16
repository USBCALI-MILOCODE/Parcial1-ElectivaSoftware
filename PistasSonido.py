import sounddevice as sd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Parámetros de grabación
duration = 5  
fs = 44100  

# Función para grabar audio
def grabar_audio(duracion, fs):
    print("Grabando...")
    audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Grabación completa")
    return audio.flatten()

# Grabar tres señales de audio (puedes grabar voces y un instrumento)
audio_1 = grabar_audio(duration, fs)
audio_2 = grabar_audio(duration, fs)
audio_3 = grabar_audio(duration, fs)

# Combinar linealmente las señales
audio_mezclado = (audio_1 + audio_2 + audio_3) / 3

# Normalización de datos
def normalizar_audio(audio):
    return (audio - np.mean(audio)) / np.std(audio)

audio_1 = normalizar_audio(audio_1)
audio_2 = normalizar_audio(audio_2)
audio_3 = normalizar_audio(audio_3)
audio_mezclado = normalizar_audio(audio_mezclado)

# Inicialización de la red neuronal ADALINE
class RedNeuronalADALINE:
    def __init__(self, entrada_size, salida_size, learning_rate=0.01):
        # Inicialización de pesos aleatorios
        self.pesos = np.random.rand(entrada_size, salida_size)
        self.learning_rate = learning_rate

    def forward(self, entrada):
        # Propagación hacia adelante
        return np.dot(entrada, self.pesos)

    def backprop(self, entrada, salida_real, salida_predicha):
        # Cálculo del error y ajuste de pesos
        error = salida_real - salida_predicha
        ajuste = self.learning_rate * np.dot(entrada.T, error) / len(entrada)
        self.pesos += ajuste

    def entrenar(self, entrada, salida_real, epochs=5000):
        for _ in range(epochs):
            salida_predicha = self.forward(entrada)
            self.backprop(entrada, salida_real, salida_predicha)

# Configuración de la red neuronal
input_size = 1  # Tamaño de la entrada
output_size = 3  # Tres señales de salida (voces y música)
red = RedNeuronalADALINE(input_size, output_size)

# Datos de entrenamiento
entrada_entrenamiento = np.array([audio_mezclado]).T  
salida_entrenamiento = np.vstack([audio_1, audio_2, audio_3]).T  

print(f"dimensiones de entrada: {entrada_entrenamiento.shape}")
print(f"dimensiones de salida: {salida_entrenamiento.shape}")

# Entrenamiento de la red
red.entrenar(entrada_entrenamiento, salida_entrenamiento, epochs=5000)

# Realizar predicciones con la red entrenada
prediccion = red.forward(entrada_entrenamiento)

# Calcular el error cuadrático medio
mse_voz1 = mean_squared_error(audio_1, prediccion[:, 0])
mse_voz2 = mean_squared_error(audio_2, prediccion[:, 1])
mse_musica = mean_squared_error(audio_3, prediccion[:, 2])

print(f"MSE Voz 1: {mse_voz1}")
print(f"MSE Voz 2: {mse_voz2}")
print(f"MSE Música: {mse_musica}")

# Graficar la señal original vs la predicha
def graficar_senal(original, prediccion, titulo):
    plt.figure(figsize=(10, 4))
    plt.plot(original, label="Original")
    plt.plot(prediccion, label="Predicción", linestyle='--')
    plt.title(titulo)
    plt.legend()
    plt.show()

# Graficar las señales
graficar_senal(audio_1, prediccion[:, 0], "Voz 1")
graficar_senal(audio_2, prediccion[:, 1], "Voz 2")
graficar_senal(audio_3, prediccion[:, 2], "Música")

# Correlación entre las señales originales y las predichas
corr_voz1, _ = pearsonr(audio_1, prediccion[:, 0])
corr_voz2, _ = pearsonr(audio_2, prediccion[:, 1])
corr_musica, _ = pearsonr(audio_3, prediccion[:, 2])

print(f"Correlación Voz 1: {corr_voz1}")
print(f"Correlación Voz 2: {corr_voz2}")
print(f"Correlación Música: {corr_musica}")
