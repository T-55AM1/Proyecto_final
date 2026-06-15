import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Cargamos y configuramos exactamente el mismo vocabulario anterior
df = pd.read_csv("datos_conversacion.csv")
tokenizer = Tokenizer(oov_token="<OOV>")
todas_las_frases = pd.concat([df['input_text'], df['target_text']])
tokenizer.fit_on_texts(todas_las_frases)

# Ajustes de dimensiones basados en los datos
X_seq = tokenizer.texts_to_sequences(df['input_text'])
Y_seq = tokenizer.texts_to_sequences(df['target_text'])
max_len = max(max(len(s) for s in X_seq), max(len(s) for s in Y_seq))

# 2. Cargamos el modelo entrenado
model = tf.keras.models.load_model("chatbot_modelo.h5")

print("\n" + "="*50)
print("   ¡TU CHATBOT ESTÁ VIVO Y LISTO PARA HABLAR!")
print("   Escribe 'salir' para cerrar la conversación.")
print("="*50 + "\n")

# 3. Bucle para chatear en tiempo real
while True:
    usuario_input = input("Tú: ")
    if usuario_input.lower() == 'salir':
        print("Chatbot: ¡Hasta luego humano!")
        break
        
    # Convertimos la frase del usuario a números y le agregamos Padding
    secuencia = tokenizer.texts_to_sequences([usuario_input])
    padded = pad_sequences(secuencia, maxlen=max_len, padding='post')
    
    # La IA genera la predicción numérica
    prediccion = model.predict(padded, verbose=0)
    
    # Traducimos los números de la predicción de vuelta a palabras legibles
    palabras_respuesta = []
    for paso in prediccion[0]:
        indice_palabra = np.argmax(paso)
        if indice_palabra == 0:
            continue
        palabra = tokenizer.index_word.get(indice_palabra, "")
        if palabra and palabra != "<OOV>":
            palabras_respuesta.append(palabra)
            
    # Unimos las palabras en una frase fluida
    respuesta_final = " ".join(palabras_respuesta)
    print(f"Chatbot: {respuesta_final}\n")
