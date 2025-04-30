#Código complementario a la práctica. Conjunto de funciones auxiliares que necesitaremos
from tensorflow.keras.preprocessing.text import Tokenizer
from prettytable import PrettyTable
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random


def carga_tokenizador(file_path):
    """
    lo que hago aquí es cargar un texto de los 3 txt del aula virtual y lo tokenizo con el tokenizador de keras
    """

    with open(file_path, "r", encoding = "utf8") as file:
        text = file.read()

    #crear un tokenizador y ajustarlo al texto
    tokenizador = Tokenizer()
    tokenizador.fit_on_texts([text])

    #crear las secuencias y el diccionario
    secuencias = tokenizador.texts_to_sequences([text])[0]
    word_index = tokenizador.word_index
    vocabulario_tamaño = len(word_index)

    print(secuencias, word_index, vocabulario_tamaño)

    return secuencias, word_index, vocabulario_tamaño

carga_tokenizador("materiales/target_words_game_of_thrones.txt")



def crear_ventana(secuencias, window_size):
    """
    Esta función crea las ventanas de contexto a partir de las secuencias de palabras.
    """
    window_size = window_size // 2 # Dividir el tamaño de la ventana entre 2 para obtener el tamaño de la ventana a la izquierda 
    #y a la derecha
    inputs = [] # lista para almacenar las palabras de entrada, es decir las palabras de contexto (las palabras vecinas)
    outputs = [] # lista para almacenar las palabras de salida, es decir la palabra objetivo
    for i in range(window_size, len(secuencias) - window_size):
        context = secuencias[i-window_size:i] + secuencias[i+1 : i+1+window_size] # palabras antes de la palabra objetivo + palabras
        #después de la palabra objetivo
        target = secuencias[i] # palabra objetivo
        inputs.append(context)
        outputs.append(target)

    return np.array(inputs), np.array(outputs)



#ME SALTO LA FUNCIÓN DE PRINT_TABLA_VENTANAS 

#recordar que el primer modelo se basa en predecir la palabra en función del contexto, utilizamos 
#tf.keras.Sequential para crear el modelo, también podríamos tf.keras.Model, pero en este caso es más sencillo
#y más legible, ya que no necesitamos crear un modelo funcional con entradas y salidas separadas. Sino que un modelo secuencial
def crear_modelo_I(neurons, vocab_size, embedding_size, window_size):
    """
    Esta función crea el modelo I (Skip-gram) para predecir la palabra objetivo a partir del contexto.
    neurons - número de neuronas en la capa oculta
    vocab_size - tamaño del vocabulario (número de palabras únicas)
    embedding_size - tamaño del vector de embedding
    window_size - tamaño de la ventana de contexto
    """
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=(window_size-1,), name='input_layer'), # -1 porque la palabra objetivo no se incluye en el contexto
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size,name='embedding_layer'), #capa de embedding, 
        #donde input_dim es el tamaño del vocabulario y output_dim es el tamaño del vector de embedding
        tf.keras.layers.Dense(units=neurons, activation='relu', name='dense_layer_0'), # Capa oculta densa aplicada a cada embedding 
        # del contexto, con dimensión "neurons"
        tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis = 1), name='average_layer'), # Se promedian los embeddings del 
        # contexto para obtener una sola representación combinada
        tf.keras.layers.Dense(units=vocab_size, activation='softmax', name='output_layer') #capa de salida, donde se predice la palabra 
        #objetivo con una función de activación softmax, que devuelve la probabilidad de cada palabra en el vocabulario
    ])

    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo


#esta va a ser una función para probar distintas combinaciones de hiperparámetros, y elegir la mejor combinación según su rendimiento 
# en el conjunto de validación. 
def comprobar_modelo_I(secuencias, vocab_size, window_size, neurons, embedding_size ):
    """
    Esta función entrena el modelo I (Skip-gram) y devuelve la precisión en el conjunto de validación. Posteriormente almacena y ordena
    los resultados y devuelve la mejor configuración de hiperparámetros.

    secuencias - secuencias de palabras tokenizadas (lista de enteros)
    vocab_size - tamaño del vocabulario (número de palabras únicas)
    window_size - tamaño de la ventana de contexto
    neurons - número de neuronas en la capa oculta
    embedding_size - tamaño del vector de embedding
    """
    results = [] # lista para almacenar los resultados de cada combinación de hiperparámetros

    #pruebo con todas las combinaciones de hiperparámetros posibles
    for window_size in window_size: 
        inputs, outputs = crear_ventana(secuencias, window_size) #crear las ventanas de contexto
        for neurons in neurons:
            for embedding_size in embedding_size:
                # la tarea de crear el modelo se vuelve trivial al haber definido la función crear_modelo_I
                modelo = crear_modelo_I(neurons, vocab_size, embedding_size, window_size) #crear el modelo
                
                #entrenar el modelo, utilizamos un valor de batch bastante grande para acelerar el entrenamiento
                history = modelo.fit(inputs, outputs, epochs=100, batch_size=1024, validation_split=0.1, verbose=False) 

                #recuperar el valor de accuracy del objeto history devuelto por el método fit 
                val_accuracy = history.history['val_accuracy'][-1] #último valor de accuracy en el conjunto de validación

                #almaceno los valores en la lista results
                results.append({
                    'neurons': neurons,
                    'embedding_size': embedding_size,
                    'window_size': window_size,
                    'val_accuracy': val_accuracy
                })

    #ordenar los resultados por val_accuracy de mayor a menor
    results = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)

    #crear una tabla para mostrar los resultados
    tabla = PrettyTable()
    tabla.field_names = ['Model','neurons', 'embedding_size', 'window_size', 'val_accuracy']
    for i, result in enumerate(results):
        tabla.add_row([f'Model {i+1}', result['neurons'], result['embedding_size'], result['window_size'], result['val_accuracy']])
    print(tabla)
    #devolver la mejor configuración de hiperparámetros
    mejor_configuracion = results[0]
    print(f"\nMejor configuración: {mejor_configuracion}. Precisión en el conjunto de validación: {mejor_configuracion['val_accuracy']}")
    return mejor_configuracion


#función extraída de materiales/embeddings_visualization.ipynb
def visualize_tsne_embeddings(words, embeddings, word_index, filename=None):
    """
    Visualizes t-SNE embeddings of selected words.

    Args:
        words (list): List of words to visualize.   
        embeddings (numpy.ndarray): Array containing word embeddings.
        word_index (dict): Mapping of words to their indices in the embeddings array.
        filename (str, optional): File to save the visualization. If None, plot is displayed.

    Returns:
        None
    """
    # Filter the embeddings for the selected words
    indices = [word_index[word] for word in words]
    selected_embeddings = embeddings[indices]

    # Set perplexity for t-SNE, it's recommended to use a value less than the number of selected words
    perplexity = min(5,len(words) - 1)

    # Use t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    # Plotting
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
        plt.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom')

    # Save or display the plot
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
