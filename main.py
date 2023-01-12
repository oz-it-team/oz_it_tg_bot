import os
import random
import requests
import telebot
import json
import openai
import io
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

#telegram api
bot = telebot.TeleBot(os.environ.get('BOT_TOKEN'))

#openai.com service
openai.api_key = os.environ.get('OPENAI_API_KEY')

#stability.ai service
stability_api = client.StabilityInference(
    key = os.environ.get('STABILITY_API_KEY'),
    verbose = True,
    engine = "stable-diffusion-512-v2-1",
)


# Command START & HELP
@bot.message_handler(commands=['help', 'start'])
def say_welcome(message):
    bot.send_message(message.chat.id,
                     'Hi, there!.\n'
                     'Мой код [тут](https://github.com/oz-it-team/oz_it_tg_bot)',
                     parse_mode='markdown')


# Command TEST
@bot.message_handler(commands=['test'])
def test(message):
    bot.send_message(message.chat.id, 'test')


# Command OPENAI QA
@bot.message_handler(commands=['openai_qa'])
def send_openai_response_to_bot(message):
    # .partition('/openai_qa ')[2] - вырезаем команду от tg из запроса
    bot.reply_to(message, get_openapi_response(message.text.partition('/openai_qa ')[2]))


# Command stability.ai: text-to-image
@bot.message_handler(commands=['generate_image'])
def send_generated_image_to_bot(message):
    query = message.text.partition('/generate_image ')[2]
    img = get_image_response(query)

    bot.reply_to(message, query)
    bot.send_photo(message.chat.id, img)


# If message send to private chat
@bot.message_handler(func=lambda message: message.chat.type == "private")
def get_private_message(message):
    bot.send_message(message.chat.id, get_answer(message.text))


# All others message
@bot.message_handler(func=lambda message: True)
def echo(message):
    print(message.text)
    if is_reply_to_bot(message):
        bot.reply_to(message, get_answer(message.text))
    elif random.randint(1, 50) == 1:
        bot.reply_to(message, get_answer(message.text))


def is_reply_to_bot(message):
    return getattr(getattr(getattr(message, 'reply_to_message', {}), 'from_user', {}), 'is_bot', False)


def get_answer(text):
    url = 'https://api.aicloud.sbercloud.ru/public/v2/boltalka/predict'
    data = {
        "instances": [
            {
                "contexts": [
                    [
                        text
                    ]
                ]
            }
        ]
    }

    response = requests.post(
        url=url,
        data=json.dumps(data),
        headers={"Content-Type": "application/json"}
    ).json()['responses']

    return response[2:len(response) - 2]

# generate answer on openai service
def get_openapi_response(text):
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=text,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        #stop=["\n"]
    )

    return response.choices[0].text

# generate img on stability.ai service
def get_image_response(text):
    answers = stability_api.generate(
        prompt=text,
        seed=992446758,
        steps=30,
        cfg_scale=8.0,
        width=512,
        height=512,
        samples=1,
        sampler=generation.SAMPLER_K_DPMPP_2M
    )

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                #img.save("image.jpg")
                
    return img

# For local testing only
if __name__ == '__main__':
    bot.infinity_polling()
