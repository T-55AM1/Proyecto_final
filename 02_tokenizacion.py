import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Iniciando la Parte 2: Procesamiento y Tokenización...")

# 1. Cargamos el archivo CSV creado en el paso anterior
df = pd.read_csv("datos_conversacion.csv")

# 2. Inicializamos el Tokenizer (asigna un número único a cada palabra diferente)
# Usamos un token especial "<OOV>" para palabras desconocidas que el bot no haya visto antes
tokenizer = Tokenizer(oov_token="<OOV>")

# Combinamos textos de entrada y salida para que el Tokenizer aprenda todo el vocabulario
todas_las_frases = pd.concat([df['input_text'], df['target_text']])
tokenizer.fit_on_texts(todas_las_frases)

# 3. Convertimos las oraciones de texto a secuencias numéricas
X_seq = tokenizer.texts_to_sequences(df['input_text'])
Y_seq = tokenizer.texts_to_sequences(df['target_text'])

# 4. Rellenamos las secuencias (Padding) para que todas tengan exactamente el mismo tamaño uniforme
X_pad = pad_sequences(X_seq, padding='post')
Y_pad = pad_sequences(Y_seq, padding='post')

print("\n¡Éxito! El texto ha sido convertido a números.")
print("\n--- Diccionario de Palabras (Muestra) ---")
# Mostramos las primeras 5 palabras asignadas a números
print(dict(list(tokenizer.word_index.items())[:5]))

print("\n--- Ejemplo de conversión ---")
print(f"Frase original: '{df['input_text'].iloc[0]}'")
print(f"En números:     {X_pad[0]}")

