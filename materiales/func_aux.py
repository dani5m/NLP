#Código complementario a la práctica. Conjunto de funciones auxiliares que necesitaremos
from tensorflow.keras.preprocessing.text import Tokenizer
from prettytable import PrettyTable
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
    vocabulario_tamaño = len(word_index) + 1

    # le sumamos uno al len(word_index) para evitar errores más tarde, por qué? porque el tokenizador para el word_index si tienes por ejemplo
    # 2024 palabras diferentes te devuelve un diccionario de 1 a 2024. Entonces, en la capa embedding (que tendrá tantas filas como vocabulario_tamaño)
    # y entendiendo que esta capa asocia a cada token un vector (word2vec), entonces si el word_index es 2024 y yo no sumo + 1 entonces el 
    # vocab_size que le pase tambien será 2024 y la matriz irá de 0 a 2023 y cuando separe las secuencias en ventanas y me aparezca el tóken
    # 2024 no lo va a encontrar en la capa embedding y me dará un error de index. En cambio si sumo +1 al tamaño del vocabulario, entonces
    # la matriz irá de 0 a 2024 y el word_index irá de 1 a 2024, y no habrá problemas de indexación para ningún token.
    return tokenizador, secuencias, word_index, vocabulario_tamaño




def crear_ventana(secuencia, window_size):
    """
    Esta función crea las ventanas de contexto a partir de las secuencias de palabras.
    """
    window_size = window_size // 2 # Dividir el tamaño de la ventana entre 2 para obtener el tamaño de la ventana a la izquierda 
    #y a la derecha
    inputs = [] # lista para almacenar las palabras de entrada, es decir las palabras de contexto (las palabras vecinas)
    outputs = [] # lista para almacenar las palabras de salida, es decir la palabra objetivo
    for i in range(window_size, len(secuencia) - window_size):
        context = secuencia[i-window_size:i] + secuencia[i+1 : i+1+window_size] # palabras antes de la palabra objetivo + palabras
        #después de la palabra objetivo
        target = secuencia[i] # palabra objetivo
        inputs.append(context)
        outputs.append(target)

    return np.array(inputs), np.array(outputs)



#ME SALTO LA FUNCIÓN DE PRINT_TABLA_VENTANAS 

#recordar que el primer modelo se basa en predecir la palabra en función del contexto, utilizamos 
#tf.keras.Sequential para crear el modelo, también podríamos tf.keras.Model, pero en este caso es más sencillo
#y más legible, ya que no necesitamos crear un modelo funcional con entradas y salidas separadas. Sino que un modelo secuencial
def crear_modelo_I(n, vocab_size, embedding_size, window_size):
    """
    Esta función crea el modelo I para predecir la palabra objetivo a partir del contexto.
    n - número de neuronas en la capa oculta
    vocab_size - tamaño del vocabulario (número de palabras únicas)
    embedding_size - tamaño del vector de embedding
    window_size - tamaño de la ventana de contexto
    """
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=(window_size-1,), name='input_layer'), # -1 porque la palabra objetivo no se incluye en el contexto
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size,name='embedding_layer'), #capa de embedding, 
        #donde input_dim es el tamaño del vocabulario y output_dim es el tamaño del vector de embedding
        tf.keras.layers.Dense(units=n, activation='relu', name='dense_layer_0'), # Capa oculta densa aplicada a cada embedding 
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
def comprobar_modelo_I(secuencias, vocab_size, window_sizes, neurons, embedding_sizes ):
    """
    Esta función entrena el modelo I y devuelve la precisión en el conjunto de validación. Posteriormente almacena y ordena
    los resultados y devuelve la mejor configuración de hiperparámetros.

    secuencias - secuencias de palabras tokenizadas (lista de enteros)
    vocab_size - tamaño del vocabulario (número de palabras únicas)
    window_size - tamaño de la ventana de contexto
    neurons - número de neuronas en la capa oculta
    embedding_size - tamaño del vector de embedding
    """
    results = [] # lista para almacenar los resultados de cada combinación de hiperparámetros

    #pruebo con todas las combinaciones de hiperparámetros posibles
    for window_size in window_sizes: 
        inputs, outputs = crear_ventana(secuencias, window_size) #crear las ventanas de contexto
        for n in neurons:
            for embedding_size in embedding_sizes:
                # la tarea de crear el modelo se vuelve trivial al haber definido la función crear_modelo_I
                modelo = crear_modelo_I(n, vocab_size, embedding_size, window_size) #crear el modelo
                
                #entrenar el modelo, utilizamos un valor de batch bastante grande para acelerar el entrenamiento
                history = modelo.fit(inputs, outputs, epochs=100, batch_size=1024, validation_split=0.1, verbose=False)  #bajo las epochs a 50 para que no tarde tanto tiempo

                #recuperar el valor de accuracy del objeto history devuelto por el método fit 
                val_accuracy = history.history['val_accuracy'][-1] #último valor de accuracy en el conjunto de validación

                #almaceno los valores en la lista results
                results.append({
                    'n': n,
                    'embedding_size': embedding_size,
                    'window_size': window_size,
                    'val_accuracy': val_accuracy
                })

    #ordenar los resultados por val_accuracy de mayor a menor
    results = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)

    #crear una tabla para mostrar los resultados
    tabla = PrettyTable()
    tabla.field_names = ['Model','n', 'embedding_size', 'window_size', 'val_accuracy']
    for i, result in enumerate(results):
        tabla.add_row([f'Model {i+1}', result['n'], result['embedding_size'], result['window_size'], result['val_accuracy']])
    print(tabla)
    #devolver la mejor configuración de hiperparámetros
    mejor_configuracion = results[0]
    print(f"\nMejor configuración: {mejor_configuracion}.")
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



def visualize_all_tsne_embeddings(embeddings, word_index, words_to_plot, words_to_label=None, filename=None):
    """
    Visualizes t-SNE embeddings of selected words with optional labeling.

    Args:
        embeddings (numpy.ndarray): Array containing word embeddings.
        word_index (dict): Mapping of words to their indices in the embeddings array.
        words_to_plot (list): List of words to plot.
        words_to_label (list, optional): List of words to label. Defaults to None.
        filename (str, optional): File to save the visualization. If None, plot is displayed.

    Returns:
        None
    """
    # Create a reverse mapping from index to word
    index_word = {index: word for word, index in word_index.items()}

    # Ensure words_to_label is a subset of words_to_plot
    if words_to_label is None:
        words_to_label = words_to_plot
    words_to_label = set(words_to_label).intersection(words_to_plot)

    # Filter the embeddings for the words to plot
    indices_to_plot = [word_index[word] for word in words_to_plot if word in word_index]
    selected_embeddings = embeddings[indices_to_plot]

    # Set perplexity for t-SNE, it's recommended to use a value less than the number of selected words
    perplexity = min(5,len(words_to_plot) - 1)

    # Use t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    reduced_embeddings = tsne.fit_transform(selected_embeddings)

    # Plotting
    plt.figure(figsize=(12, 12))
    for i, index in enumerate(indices_to_plot):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], alpha=0.5)
        if index_word[index] in words_to_label:  # Annotate only selected words
            plt.annotate(index_word[index],
                         xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')


# Función para cargar las palabras target
def palabras_target(ruta):
    with open(ruta, "r") as file:
        # Leer el archivo
        words = file.readlines()

    # Borrar espacios
    words = [word.strip() for word in words]

    return words


# Función para calcular la similitud del coseno entre embeddings de palabras
def calcular_similitud_coseno(words, word_index, embeddings, titulo):

    # Diccionario donde se guardarán las palabras más similares a cada palabra objetivo
    word_proximities = {}

    # Iterar por cada palabra objetivo
    for word in words:
        # Obtener el índice de la palabra en el vocabulario
        index = word_index[word]
        # Obtener su vector embedding
        embedding = embeddings[index]

        # Calcular la similitud del coseno entre el embedding de la palabra y todos los embeddings
        similarity = cosine_similarity([embedding], embeddings)[0].tolist()

        # Obtener los índices ordenados de mayor a menor similitud
        indices = np.argsort(similarity)[::-1]
        # Eliminar el índice de la palabra original y quedarse con los 10 más similares
        indices = [i for i in indices if i != index][:10]

        # Obtener las palabras correspondientes a esos índices
        palabras_proximas = [list(word_index.keys())[i] for i in indices]

        # Guardar las palabras más próximas en el diccionario
        word_proximities[word] = palabras_proximas

    # Crear tabla con los resultados
    table = PrettyTable()
    table.field_names = ['Target', 'Palabras máis próximas ' + titulo]

    # Añadir filas a la tabla
    for word, nearby_words in word_proximities.items():
        table.add_row([word, ', '.join(nearby_words)])

    # Mostrar la tabla
    print(table)




# Función que genera pares (target, contexto) positivos y negativos para entrenamiento 
def cartesian_product(sequences):
    # Tamaño de la ventana de contexto: 5 // 2 = 2 (contexto a izquierda y derecha)
    n = 2

    # Listas donde se almacenarán los pares de entrada y sus etiquetas
    input_pairs = []
    output = []

    # Iterar por cada palabra con suficiente contexto a izquierda y derecha
    for i in range(n, len(sequences) - n):
        # Obtener las palabras del contexto (ventana a izquierda y derecha, excluyendo la palabra central)
        context = sequences[i - n:i] + sequences[i + 1:i + n + 1]
        # Palabra objetivo (central en la ventana)
        target_word = sequences[i]

        # Para cada palabra del contexto, crear un par positivo con la palabra objetivo
        for word in context:
            input_pairs.append([target_word, word])  # Par (target, contexto)
            output.append(1)  # Etiqueta positiva

        # Generar un ejemplo negativo: una palabra aleatoria que no está en el contexto
        random_number = random.randint(0, len(sequences) - 1)
        while random_number in context:
            random_number = random.randint(0, len(sequences) - 1)

        # Añadir el par negativo (target, palabra_aleatoria)
        input_pairs.append([target_word, random_number])
        output.append(0)  # Etiqueta negativa

    # Convertir las listas a arrays de NumPy y devolverlas
    return np.array(input_pairs), np.array(output)


#A diferencia del modelo I, en este modelo nos hemos decantado por usar el modelo funcional de keras

def crear_modelo_II(vocab_size, embedding_size):

    input_layer1 = tf.keras.Input(shape=(1,), name='input_layer1')
    input_layer2 = tf.keras.Input(shape=(1,), name='input_layer2')

    embedding_layer1 = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, name='embedding_layer1')

    embedding1 = embedding_layer1(input_layer1)
    embedding2 = embedding_layer1(input_layer2)

    dot_product = tf.keras.layers.Dot(axes=2)([embedding1, embedding2])

    flatten = tf.keras.layers.Flatten()(dot_product)

    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)

    model = tf.keras.Model(inputs=[input_layer1, input_layer2], outputs=output_layer)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


