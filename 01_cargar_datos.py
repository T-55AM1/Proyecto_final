import pandas as pd

print("Iniciando desde el principio: Creando base de datos interna...")

# Creamos el conjunto de datos conversacionales directamente en el código
conversaciones = {
    "input_text": [
        "hola", 
        "como estas?", 
        "que haces?", 
        "necesito ayuda", 
        "mi computadora no enciende", 
        "gracias por la ayuda",
        "adios"
    ],
    "target_text": [
        "hola humano, en que puedo ayudarte?", 
        "soy un robot, no tengo sentimientos pero funciono al 100%", 
        "estoy procesando datos para aprender a hablar contigo", 
        "claro que si, dime que problema tienes hoy", 
        "revisa si esta conectada a la corriente electrica o intenta mantener presionado el boton de encendido", 
        "de nada, es un placer ayudarte",
        "hasta luego, que tengas un gran dia!"
    ]
}

# Convertimos los datos a una tabla (DataFrame)
df_limpio = pd.DataFrame(conversaciones)

# Guardamos los datos localmente en un archivo CSV limpio
df_limpio.to_csv("datos_conversacion.csv", index=False, encoding="utf-8")

print(f"¡Éxito total! Base de datos local creada con {len(df_limpio)} interacciones.")
print("\n--- Vista previa de los datos de entrenamiento ---")
print(df_limpio.head(3))
