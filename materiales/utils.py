# Prácticas Módulo II PLE. Raquel Piñeiro Pereira. Xairo Campos Blanco.
# Este é un ficheiro coas funcións que se chaman dende o Jupyter Notebook

# Importar librerías que se utilizarán
from keras.preprocessing.text import Tokenizer
import numpy as np
from prettytable import PrettyTable
import tensorflow as tf
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random




# Función para cargar os textos e tokenizalos
def cargar_tokenizar(file_path):

    # Abrir o texto
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Crear un tokenizador e axustalo
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])

    # Crear as secuencias e o diccionario word_index
    sequences = tokenizer.texts_to_sequences([text])[0]
    word_index = tokenizer.word_index
    vocab_size = len(word_index)

    # Devolver todo
    return tokenizer, word_index, sequences, vocab_size


# Función para crear ventanas. Esta función xa devolve unha lista de input_pairs e outra de output_words
def crear_ventana(sequences, n):
    n = n // 2
    input_pairs = []
    output_words = []
    for i in range(n, len(sequences) - n):
        context = sequences[i - n:i] + sequences[i + 1:i + n + 1]
        target_word = sequences[i]
        input_pairs.append(context)
        output_words.append(target_word)

    return np.array(input_pairs), np.array(output_words)


# Función para imprimir nunha táboa as ventanas que se crean(só as primeiras 5)
def print_tabla_ventanas(input_pairs, output_words, tokenizer, titulo):
    table = PrettyTable(["ID", "Input - " + titulo, "Output"])

    # Iterar sobre a función para mostrar as ventanas creadas
    for idx, (input_pair, output_word) in enumerate(zip(input_pairs, output_words)):
        if idx >= 5:
            break
        central_word = tokenizer.index_word[output_word]
        context_words = [tokenizer.index_word[idx] for idx in input_pair]
        table.add_row([idx, context_words, central_word])

    print(table)


# Función para crear o 1º modelo. Esta función serve para crear modelos con distintos window_size, vocab_size,.. 
# Axústase a cada texto e á función de probar modelos
def crear_modelo_I(neurons, vocab_size, embedding_size, window_size):
    
    # capa de entrada. Window_size-1 polo que se explicou de que aínda que a ventana teña tamaño n a entrada é n-1
    input_layer = tf.keras.Input(shape=(window_size-1,), name='input_layer')

    # capa de embedding
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, name='embedding_layer')(input_layer)

    # Capa dense
    dense_layer_0 = tf.keras.layers.Dense(units=neurons, activation='relu', name='dense_layer_0')(embedding_layer)

    # Capa lambda para o average
    average_layer = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis = 1), name='average_layer')(dense_layer_0)

    # Capa de saída
    output_layer = tf.keras.layers.Dense(units=vocab_size, activation='softmax', name='output_layer')(average_layer)

    # Crear modelo
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model





# Función para probar distintos hiperparámetros ata atopar o mellor
def chequear_modelo_I(secuencia, vocab_size, window_sizes, units_values, embedding_sizes):

    # Inicializar estas variables para almacenar os mellores resultados
    results = []
    cnt = 0
    
    # Barra de progreso
    total_iterations = len(window_sizes) * len(units_values) * len(embedding_sizes)
    progress_bar = tqdm(total=total_iterations, desc='Progress', unit='model')

    # Iterar sobre todos os parámetros propostos
    for window_size in window_sizes:

        # Recalcular input_pairs e output_words para o tamaño de ventana desta iteración
        input_pairs, output_words = crear_ventana(secuencia, window_size)       

        # Iterar sobre o resto de hiperparámetros
        for units in units_values:
            for embedding_size in embedding_sizes:
                
                # Construir o modelo. Debido a como se fixo a función de crear modelos, é moi sinxelo
                model = crear_modelo_I(units, vocab_size, embedding_size, window_size)
                
                # Entrenar o modelo
                history = model.fit(input_pairs, output_words, epochs=100, validation_split=0.1, batch_size = 1024, verbose=False)
                
                # Recuperar o validation accuracy do modelo
                val_acc = history.history['val_accuracy'][-1]
                
                # Almacenar resultados
                results.append({'window_size': window_size, 'units': units, 'embedding_size': embedding_size, 'val_accuracy': val_acc})
                cnt += 1    

                # Barra progreso
                progress_bar.update(1)

    # Barra progreso
    progress_bar.close()


    # Ordenar resultados por val_accuracy para ver mellor modelo
    resultados_ordenados = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)

    # Crear tabla para ver os resultados ordenados
    table = PrettyTable()
    table.field_names = ['Modelo', 'Window Size🪟', 'Units🧠', 'Embedding Size🔖', 'Validation Accuracy🔬']
    for i, result in enumerate(resultados_ordenados, start=1):
        table.add_row([f"Model {i}", result['window_size'], result['units'], result['embedding_size'], result['val_accuracy']])
    print(table)

    # Gardar a mellor configuración
    best_model_config = resultados_ordenados[0]
    best_acc = best_model_config["val_accuracy"]
    print(f"\n🥇Mellor modelo: {best_model_config}. Validation Accuracy🔬: {best_acc}")

    # Devolver a mellor configuración para poder entrenar
    return best_model_config







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


# Función dada nos materiais modificada para imprimir os embeddings antes e despois
def visualize_all_tsne_embeddings(embeddings_first, embeddings_second, word_index, words_to_plot, titulo, words_to_label=None, filename=None):
    """
    Visualizes t-SNE embeddings of selected words with optional labeling for two sets of embeddings.

    Args:
        embeddings_first (numpy.ndarray): Array containing word embeddings for the first set.
        embeddings_second (numpy.ndarray): Array containing word embeddings for the second set.
        word_index (dict): Mapping of words to their indices in the embeddings arrays.
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
    selected_embeddings_first = embeddings_first[indices_to_plot]
    selected_embeddings_second = embeddings_second[indices_to_plot]

    # Set perplexity for t-SNE, it's recommended to use a value less than the number of selected words
    perplexity = min(5, len(words_to_plot) - 1)

    # Use t-SNE to reduce dimensionality for the first set of embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
    reduced_embeddings_first = tsne.fit_transform(selected_embeddings_first)

    # Use t-SNE to reduce dimensionality for the second set of embeddings
    reduced_embeddings_second = tsne.fit_transform(selected_embeddings_second)

    # Plotting
    plt.figure(figsize=(20, 20))

    # Plot for the first set of embeddings
    plt.subplot(1, 2, 1)
    for i, index in enumerate(indices_to_plot):
        plt.scatter(reduced_embeddings_first[i, 0], reduced_embeddings_first[i, 1], alpha=0.5)
        if index_word[index] in words_to_label:
            plt.annotate(index_word[index],
                         xy=(reduced_embeddings_first[i, 0], reduced_embeddings_first[i, 1]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
    plt.title('Embeddings Antes')
    plt.suptitle(f'Visualización t-SNE {titulo}')

    # Repetir para o segundo subplot
    plt.subplot(1, 2, 2)
    for i, index in enumerate(indices_to_plot):
        plt.scatter(reduced_embeddings_second[i, 0], reduced_embeddings_second[i, 1], alpha=0.5)
        if index_word[index] in words_to_label:
            plt.annotate(index_word[index],
                         xy=(reduced_embeddings_second[i, 0], reduced_embeddings_second[i, 1]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
    plt.title('Embeddings Despois')

    plt.show()
            





# Función para cargar as palabras target(as que se van comprobar no mapa)
def palabras_target(ruta):
    with open(ruta, "r") as file:
        # Ler o ficheiro
        words = file.readlines()

    # Borrar espacios
    words = [word.strip() for word in words]

    return words



# función para calcular a similutde do coseno
def calcular_cosine(words, word_index, embeddings_despois, titulo):
    # Iniciarlizar diccionario vacío
    word_proximities = {}

    for word in words:
        index = word_index[word]
        embedding = embeddings_despois[index]

        similarity = cosine_similarity([embedding], embeddings_despois)[0].tolist()

        # Buscar os 10 resultados máis altos(máis similares)
        indices = np.argsort(similarity)[-1:-11:-1]
        indices = [x - 1 for x in indices]

        # Recuperar cales son as palabras máis próximas(nas liñas anteriores o que se recuperan son os índices)
        palabras_proximas = [list(word_index.keys())[i] for i in indices]

        # Gardar as palabras nun diccionario
        word_proximities[word] = palabras_proximas

    # Mostrar nunha táboa
    table = PrettyTable()
    table.field_names = ['Target', 'Palabras máis próximas ' + titulo]
    for word, nearby_words in word_proximities.items():
        table.add_row([word, ', '.join(nearby_words)])

    # Mostrar táboa
    print(table)


# Producto cartesiano(da palabra obxectivo co resto)
def cartesian_product(sequences):
    n = 5 // 2
    input_pairs = []
    output = []
    for i in range(n, len(sequences) - n):
        context = sequences[i - n:i] + sequences[i + 1:i + n + 1]
        target_word = sequences[i]
        for word in context:
            input_pairs.append([target_word, word])
            output.append(1)

        random_number = random.randint(0, len(sequences) - 1)
        while random_number in context:
            random_number = random.randint(0, len(sequences) - 1)
        input_pairs.append([target_word, random_number])
        output.append(0)

    return np.array(input_pairs), np.array(output)










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









# Neutralizado. Función que intenta eliminar os sesgos de xénero
def neutralize(embeddings, word_index, target_word):

    # recuperar o índice e o embedding da target(da que se quere neutralizar os sesgos)
    target_word_index = word_index[target_word]
    target_embedding = embeddings[target_word_index]

    # Para neutralizar o sesgo réstaselle á palabra o producto vectorial do seu embedding pola "dirección de xénero".
    # Esta dirección de xénero pode calcularse de moitas formas distintas, pero unha aproximación sinxela é restar o embedding de home menos o de muller
    # Cálculo da dirección de xénero
    gender_direction = embeddings[word_index["man"]] - embeddings[word_index["woman"]]
    neutralized_embedding = target_embedding - np.dot(target_embedding, gender_direction) * gender_direction

    # Actualizar a lista de embeddings co novo embedding neutralizado
    embeddings[word_index[target_word]] = neutralized_embedding

    # Devolver o embedding neutralizado
    return embeddings