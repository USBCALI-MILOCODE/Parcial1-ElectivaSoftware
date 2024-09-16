import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        """
        Inicializa el perceptrón con los pesos, el sesgo y la tasa de aprendizaje.

        Args:
            input_size (int): Número de entradas de la red neuronal.
            output_size (int): Número de salidas de la red neuronal.
            learning_rate (float): Tasa de aprendizaje para el ajuste de los pesos. Default es 0.1.
        """
        self.weights = np.random.rand(input_size, output_size) * 0.01  # Inicializa los pesos aleatoriamente.
        self.bias = np.zeros((1, output_size))  # Inicializa el sesgo con ceros.
        self.learning_rate = learning_rate  # Tasa de aprendizaje.
        self.errors = []  # Lista para almacenar los errores en cada época.

    def sigmoid(self, z):
        """
        Función de activación sigmoide.

        Args:
            z (numpy.ndarray): Valor de entrada.

        Returns:
            numpy.ndarray: Valor transformado por la función sigmoide.
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """
        Derivada de la función de activación sigmoide.

        Args:
            z (numpy.ndarray): Valor de entrada.

        Returns:
            numpy.ndarray: Derivada de la función sigmoide aplicada a z.
        """
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        """
        Propagación hacia adelante.

        Args:
            X (numpy.ndarray): Datos de entrada.

        Returns:
            numpy.ndarray: Salida de la red neuronal.
        """
        self.z = np.dot(X, self.weights) + self.bias  # Cálculo de la entrada ponderada.
        return self.sigmoid(self.z)  # Aplicación de la función de activación.
    
    def backward(self, X, y, output):
        """
        Propagación hacia atrás y ajuste de pesos y sesgo.

        Args:
            X (numpy.ndarray): Datos de entrada.
            y (numpy.ndarray): Salida deseada.
            output (numpy.ndarray): Salida actual de la red.
        """
        error = y - output  # Error entre la salida deseada y la salida actual.
        self.errors.append(np.mean(np.square(error)))  # Cálculo y almacenamiento del error cuadrático medio.
        d_weights = np.dot(X.T, error * self.sigmoid_derivative(self.z))  # Gradiente de los pesos.
        d_bias = np.sum(error * self.sigmoid_derivative(self.z), axis=0, keepdims=True)  # Gradiente del sesgo.
        self.weights += self.learning_rate * d_weights  # Actualización de los pesos.
        self.bias += self.learning_rate * d_bias  # Actualización del sesgo.

    def plot_errors(self):
        """
        Grafica la evolución del error cuadrático medio durante el entrenamiento.
        """
        plt.plot(self.errors)
        plt.xlabel('Épocas')
        plt.ylabel('Error cuadrático medio')
        plt.title('Evolución del Error Durante el Entrenamiento')
        plt.show()

    def train(self, X, y, epochs=1000):
        """
        Entrena la red neuronal.

        Args:
            X (numpy.ndarray): Datos de entrada.
            y (numpy.ndarray): Salida deseada.
            epochs (int): Número de épocas de entrenamiento. Default es 1000.

        Returns:
            numpy.ndarray: Salida final de la red neuronal después del entrenamiento.
        """
        for epoch in range(epochs):
            output = self.forward(X)  # Propagación hacia adelante.
            self.backward(X, y, output)  # Propagación hacia atrás.
            if epoch % 100 == 0:  # Mostrar el error cada 100 épocas.
                print(f'Época {epoch}: Error cuadrático medio = {self.errors[-1]}')
        self.plot_errors()  # Mostrar la gráfica de los errores.
        return output
    
    def predict(self, X):
        """
        Realiza una predicción para la entrada X.

        Args:
            X (numpy.ndarray): Datos de entrada.

        Returns:
            numpy.ndarray: Salida de la red neuronal.
        """
        return self.forward(X)

def definir_matrices():
    """
    Define las representaciones de las letras en matrices 5x5 usando bits.

    Returns:
        dict: Diccionario con las letras y sus matrices correspondientes.
    """
    letras = {
    'A': np.array([[0, 1, 1, 1, 0],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1]]),
    'B': np.array([[1, 1, 1, 1, 0],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 0],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 0]]),
    'C': np.array([[0, 1, 1, 1, 1],
                   [1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1]]),
    'D': np.array([[1, 1, 1, 1, 0],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 0]]),
    'E': np.array([[1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 0],
                   [1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1]]),
    'F': np.array([[1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 0],
                   [1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0]]),
    'G': np.array([[0, 1, 1, 1, 0],
                   [1, 0, 0, 0, 0],
                   [1, 0, 1, 1, 0],
                   [1, 0, 0, 0, 1],
                   [0, 1, 1, 1, 0]]),
    'H': np.array([[1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1]]),
    'I': np.array([[0, 1, 1, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, 1, 1, 0]]),
    'J': np.array([[0, 1, 1, 1, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 0],
                   [1, 0, 0, 1, 0],
                   [0, 1, 1, 1, 0]]),
    'K': np.array([[1, 0, 0, 1, 0],
                   [1, 0, 1, 0, 0],
                   [1, 1, 0, 0, 0],
                   [1, 0, 1, 0, 0],
                   [1, 0, 0, 1, 0]]),
    'L': np.array([[1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1]]),
    'M': np.array([[1, 0, 0, 0, 1],
                   [1, 1, 0, 1, 1],
                   [1, 0, 1, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1]]),
    'N': np.array([[1, 0, 0, 0, 1],
                   [1, 1, 0, 0, 1],
                   [1, 0, 1, 0, 1],
                   [1, 0, 0, 1, 1],
                   [1, 0, 0, 0, 1]]),
    'O': np.array([[0, 1, 1, 1, 0],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [0, 1, 1, 1, 0]]),
    'P': np.array([[1, 1, 1, 1, 0],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 0],
                   [1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0]]),
    'Q': np.array([[1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 1]]),
    'R': np.array([[1, 1, 1, 1, 0],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 0],
                   [1, 0, 1, 0, 0],
                   [1, 0, 0, 1, 0]]),
    'S': np.array([[0, 1, 1, 1, 1],
                   [1, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 1],
                   [1, 1, 1, 1, 0]]),
    'T': np.array([[1, 1, 1, 1, 1],
                   [0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0]]),
    'U': np.array([[1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 1, 1, 1, 1]]),
    'V': np.array([[1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [0, 1, 0, 1, 0],
                   [0, 0, 1, 0, 0]]),
    'W': np.array([[1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [1, 0, 1, 0, 1],
                   [1, 1, 0, 1, 1],
                   [1, 0, 0, 0, 1]]),
    'X': np.array([[1, 0, 0, 0, 1],
                   [0, 1, 0, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, 0, 1, 0],
                   [1, 0, 0, 0, 1]]),
    'Y': np.array([[1, 0, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [0, 1, 0, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 0]]),
    'Z': np.array([[1, 1, 1, 1, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, 0, 0, 0],
                   [1, 1, 1, 1, 1]])
  }
    return letras

def convertir_a_vector(matriz):
    """
    Convierte una matriz 5x5 en un vector de 25 elementos.

    Args:
        matriz (numpy.ndarray): Matriz 5x5.

    Returns:
        numpy.ndarray: Vector de 25 elementos.
    """
    return matriz.flatten()

def definir_datos(letras):
    """
    Define los datos de entrada y salida a partir de las matrices de letras.

    Args:
        letras (dict): Diccionario de letras y sus matrices.

    Returns:
        tuple: Datos de entrada X y salidas deseadas y.
    """
    X = np.array([convertir_a_vector(letras[letra]) for letra in sorted(letras)])
    y = np.array([
    [0, 0, 0, 0, 1],  # A
    [0, 0, 0, 1, 0],  # B
    [0, 0, 0, 1, 1],  # C
    [0, 0, 1, 0, 0],  # D
    [0, 0, 1, 0, 1],  # E
    [0, 0, 1, 1, 0],  # F
    [0, 0, 1, 1, 1],  # G
    [0, 1, 0, 0, 0],  # H
    [0, 1, 0, 0, 1],  # I
    [0, 1, 0, 1, 0],  # J
    [0, 1, 0, 1, 1],  # K
    [0, 1, 1, 0, 0],  # L
    [0, 1, 1, 0, 1],  # M
    [0, 1, 1, 1, 0],  # N
    [0, 1, 1, 1, 1],  # O
    [1, 0, 0, 0, 0],  # P
    [1, 0, 0, 0, 1],  # Q
    [1, 0, 0, 1, 0],  # R
    [1, 0, 0, 1, 1],  # S
    [1, 0, 1, 0, 0],  # T
    [1, 0, 1, 0, 1],  # U
    [1, 0, 1, 1, 0],  # V
    [1, 0, 1, 1, 1],  # W
    [1, 1, 0, 0, 0],  # X
    [1, 1, 0, 0, 1],  # Y
    [1, 1, 0, 1, 0]   # Z
])
    return X, y


def introducir_ruido(X, porcentaje_ruido=0.1):
    """
    Introduce ruido aleatorio en los datos de entrada.

    Args:
        X (numpy.ndarray): Datos de entrada.
        porcentaje_ruido (float): Porcentaje de valores que serán alterados por ruido.

    Returns:
        numpy.ndarray: Datos de entrada con ruido añadido.
    """
    X_ruidoso = X.copy()  # Crea una copia de los datos de entrada para modificarla.
    num_muestras, num_features = X.shape  # Obtiene las dimensiones de los datos.

    # Calcula el número total de valores que se alterarán.
    num_valores_ruido = int(num_muestras * num_features * porcentaje_ruido)
    
    # Genera índices aleatorios para alterar los valores.
    indices = np.random.choice(num_muestras * num_features, num_valores_ruido, replace=False)
    
    # Alterna los valores en los índices seleccionados.
    for index in indices:
        fila = index // num_features
        columna = index % num_features
        X_ruidoso[fila, columna] = 1 - X_ruidoso[fila, columna]

    return X_ruidoso

def imprimir_matriz_como_texto(matriz):
    """
    Imprime la matriz en formato legible como texto.

    Args:
        matriz (numpy.ndarray): La matriz a imprimir, esperada en formato 5x5.
    """
    for fila in matriz:
        print(' '.join(map(str, fila)))
    print()

def mostrar_matrices_con_ruido(letras, porcentaje_ruido):
    """
    Muestra las matrices de las letras originales y con ruido en la consola.

    Args:
        letras (dict): Diccionario de letras y sus matrices.
        porcentaje_ruido (float): Porcentaje de ruido a agregar a las matrices.
    """
    for letra, matriz in letras.items():
        matriz_ruidosa = introducir_ruido(matriz, porcentaje_ruido)

        # Imprimir la matriz con ruido
        print(f"{letra} con Ruido:")
        imprimir_matriz_como_texto(matriz_ruidosa.reshape(5, 5))

def main():
    letras = definir_matrices()
    X, y = definir_datos(letras)
    
    perceptron = Perceptron(input_size=25, output_size=5)
    
    print("Entrenando la red neuronal...")
    perceptron.train(X, y)
    
     # Introducir ruido y reentrenar
    X_noisy = introducir_ruido(X, porcentaje_ruido=0.1)

    print("Entrenando la red con datos ruidosos...")
    perceptron.train(X_noisy, y)

    print("Resultados después del entrenamiento:")
    for letra, matriz in letras.items():
        vector = convertir_a_vector(matriz)
        prediccion = perceptron.predict(vector.reshape(1, -1))
        print(f"Letra {letra}:")
        imprimir_matriz_como_texto(matriz)
        print(f"Entrada (vector): {vector}")
        print(f"Predicción (valores sigmoides): {prediccion.flatten()}")
        print(f"Predicción (código binario): {np.round(prediccion.flatten()).astype(int)}\n")
    
   
   

    # Mostrar las matrices originales y con ruido
    mostrar_matrices_con_ruido(letras, porcentaje_ruido=0.1)

if __name__ == "__main__":
    main()
