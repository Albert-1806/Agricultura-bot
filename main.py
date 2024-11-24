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