# Proyecto de Detección de Estado de Plantas con Discord Bot y Teachable Machine

Este modelo permite detectar el estado de una planta utilizando un modelo de Teachable Machine integrado en un bot de Discord.

Instrucciones Paso a Paso
-
1. Exporta tu modelo de Teachable Machine

   - Exporta tu propio modelo de Teachable Machine en formato Keras.

   - Guarda los archivos **keras_model.h5** y **labels.txt.**

   - Asegúrate de que estos archivos reflejan el entrenamiento de tu modelo personalizado.

2. Configurar el Visual Studio Code

 a) Instalar Python 3.10

   - Descarga Python 3.10 desde https://www.python.org/downloads/release/python-3100/ 

   - Nota: No es necesario desinstalar Python 3.12 si ya lo tienes instalado.

 b) Verificar las versiones de Python

   - En la terminal, ejecuta el siguiente comando para verificar que ambas 
      versiones están instaladas:

    py -op 
 c) Crear el entorno virtual

   - En Visual Studio Code, ejecuta los siguientes comandos:

    pip install pipenv
    pipenv --python "C:\Users\<tu_usuario>\AppData\Local\Programs\Python\Python310\python.exe"

    Nota: Puedes encontrar la ruta de Python buscando en tu sistema archivos o usando py -0p.

 d) Activar el entorno virtual

   - Ejecuta en la terminal:

    pipenv shell

3. Instalar Dependencias

 a) En la terminal, ejecuta las siguientes dependencias:

    pipenv install Pillow==9.1.0
    pipenv install tensorflow==2.8.0
    pipenv install discord.py
    pipenv install numpy==1.21.6
    pipenv install keras
    pipenv install requests
    pipenv install protobuf==3.20.3

4. Crear los Archivos del Proyecto
 a) Archivo model.py

Crea un archivo **model.py** y pega el siguiente código que servirá para cargar tu modelo de Keras y hacer las predicciones. Este archivo se puede usar con cualquier modelo que hayas exportado desde Teachable Machine:

    from keras.models import load_model  # TensorFlow is required for Keras to work
    from PIL import Image, ImageOps  # Install pillow instead of PIL
    import numpy as np

    def get_class(model_path, labels_path, image_path):
        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Load the model
        model = load_model(model_path, compile=False)

        # Load the labels
        class_names = open(labels_path, "r").readlines()

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open(image_path).convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        return (class_name[2:],confidence_score)

 b) Archivo main.py

Crea el archivo **main.py** para tu bot de Discord, que usará tu modelo para hacer predicciones basadas en imágenes que se suben al canal de Discord:

    import discord
    from discord.ext import commands
    import os, random
    import requests

    from model import get_class 

    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='!', intents=intents)

    @bot.event
    async def on_ready():
        print(f'We have logged in as {bot.user}')

    @bot.command()
    async def hello(ctx):
        await ctx.send(f'Hi! I am a bot {bot.user}!')

    @bot.command()
    async def heh(ctx, count_heh = 5):
        await ctx.send("he" * count_heh)

    @bot.command()
    async def save(ctx):
        if ctx.message.attachments:
            for attachment in ctx.message.attachments:
                file_name = attachment.filename
                file_url = attachment.url
                await attachment.save(f"./img/{file_name}")
                await ctx.send(f"Guarda la imagen en ./img/{file_url}")

        else:
            await ctx.send("No se ha podido cargar la imagen :(")

    @bot.command()
    async def check(ctx):
        if ctx.message.attachments:
            for attachment in ctx.message.attachments:
                file_name = attachment.filename
                class_name, confidence_score = get_class(model_path="keras_model.h5", labels_path="labels.txt", image_path=f"./img/{file_name}")
                response_message = f"**Predicción:** {class_name}\n**Confianza:** {confidence_score:.2f}"
                await ctx.send(response_message)
        else:
            await ctx.send("No se ha podido revisar la imagen :(")

        bot.run("TU_TOKEN_DISCORD") 

5. Ejecutar el Proyecto

    - Sube tu modelo **keras_model.h5** y el archivo **labels.txt** a la carpeta del proyecto.

    - Ejecuta tu bot con : 
     
          python main.py

    - Sube una imagen en Discord y utiliza el comando !save para guardar la imagen y consecutivo usa !check para compararlo. 

Adicional:
    Asegúrate de modificar **TU_TOKEN_DISCORD** con el token de tu bot de Discord.
