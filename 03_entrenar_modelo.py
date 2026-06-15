import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

print("Iniciando la Parte 3: Construcción y Entrenamiento del Chatbot...")

# 1. Recargamos los datos y configuramos el Tokenizer tal como en el paso anterior
df = pd.read_csv("datos_conversacion.csv")
tokenizer = Tokenizer(oov_token="<OOV>")
todas_las_frases = pd.concat([df['input_text'], df['target_text']])
tokenizer.fit_on_texts(todas_las_frases)

vocab_size = len(tokenizer.word_index) + 1

# Convertimos texto a secuencias numéricas fijas
X_seq = tokenizer.texts_to_sequences(df['input_text'])
Y_seq = tokenizer.texts_to_sequences(df['target_text'])

# Encontramos la longitud máxima para que todas las secuencias midan lo mismo
max_len = max(max(len(s) for s in X_seq), max(len(s) for s in Y_seq))
X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post')
Y_pad = pad_sequences(Y_seq, maxlen=max_len, padding='post')

# Convertimos las respuestas a un formato que la última capa de la red entienda (One-Hot Encoding)
Y_final = tf.keras.utils.to_categorical(Y_pad, num_classes=vocab_size)

# 2. DEFINICIÓN DE LA RED NEURONAL (Misma filosofía del tutorial Seq2Seq simplificado)
model = Sequential([
    # Capa de incrustación: convierte números enteros en vectores densos con significado semántico
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    # Capa LSTM: recuerda el orden y el contexto de las palabras en una oración
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    # Capa Densa: decide matemáticamente cuál es la palabra más probable para responder
    Dense(vocab_size, activation='softmax')
])

# Compilamos el cerebro del robot
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- ¡Estructura de la IA construida con éxito! ---")
model.summary()

print("\nEntrenando al chatbot durante 100 iteraciones (epochs)... Por favor espera.")
# Entrenamos el modelo con nuestros vectores numéricos
model.fit(X_pad, Y_final, epochs=100, verbose=1)

# Guardamos el cerebro entrenado y las palabras para el paso final
model.save("chatbot_modelo.h5")
print("\n¡Éxito! El modelo ha sido entrenado y guardado como 'chatbot_modelo.h5'.")

