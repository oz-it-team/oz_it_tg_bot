import io
import json
import logging
import os
import random
import warnings

import openai
import requests
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import telebot
from PIL import Image
from langchain_community.chat_models import ChatYandexGPT
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from stability_sdk import client

# telegram api
bot = telebot.TeleBot(os.environ.get('BOT_TOKEN'))

yandex = ChatYandexGPT(api_key=os.environ.get('YA_API_KEY'), folder_id=os.environ.get('YA_FOLDER_ID'),
                       model='yandexgpt-lite')

logging.getLogger().setLevel(logging.DEBUG)

# openai.com service
openai.api_key = os.environ.get('OPENAI_API_KEY')

# stability.ai service
stability_api = client.StabilityInference(
    key=os.environ.get('STABILITY_API_KEY'),
    verbose=True,
    engine="stable-diffusion-512-v2-1",
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

    bot.reply_to(message, "Лови 🎨")
    bot.send_photo(message.chat.id, img)


@bot.message_handler(commands=['yandex'])
def send_generated_image_to_bot(message):
    bot.reply_to(message, ask_yandex_gpt_rest(create_yandex_gpt_messages(' '.join(message.text.split()[1:]))),
                 parse_mode='markdown')


# If message send to private chat
@bot.message_handler(func=lambda message: message.chat.type == "private")
def get_private_message(message):
    bot.send_message(message.chat.id, ask_yandex_gpt_rest(create_yandex_gpt_messages(message.text)),
                     parse_mode='markdown')


# All others message
@bot.message_handler(func=lambda message: True)
def echo(message):
    print(message.text)
    if is_reply_to_bot(message):
        bot.reply_to(
            message,
            ask_yandex_gpt_rest(create_yandex_gpt_messages_with_reply(message.reply_to_message.text, message.text)),
            parse_mode='markdown'
        )
    elif random.randint(1, 30) == 1:
        bot.reply_to(message, ask_yandex_gpt_rest(create_yandex_gpt_messages(message.text)), parse_mode='markdown')


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
        # stop=["\n"]
    )

    return response.choices[0].text


# generate img on stability.ai service
def get_image_response(text):
    answers = stability_api.generate(
        prompt=text,
        seed='',
        steps=50,
        cfg_scale=12.0,
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
                # img.save("image.jpg")

    return img


def ask_yandex_gpt(text):
    print('Asking yaGpt about:' + text)
    messages = get_gpt_system_role()
    messages.append(HumanMessage(content=text))
    answer = yandex.invoke(messages)
    print('Answer yaGpt:' + answer.content)
    return answer.content


def get_gpt_system_role() -> list[BaseMessage]:
    return [
        SystemMessage(
            content="Ты лучший ии-ассистент со знанием программирования. Говори от женского имени."
        )
    ]


def ask_yandex_gpt_rest(messages):
    url = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'
    model = 'gpt://' + os.environ.get('YA_FOLDER_ID') + '/yandexgpt-lite'
    data = {
        "modelUri": model,
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": "1000"
        },
        "messages": messages
    }

    response = requests.post(
        url=url,
        data=json.dumps(data),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Api-Key " + os.environ.get('YA_API_KEY'),
            "x-folder-id": os.environ.get('YA_FOLDER_ID')
        }
    ).json()

    return response['result']['alternatives'][0]['message']['text']


def create_yandex_gpt_messages(text):
    print('Asking yaGpt-rest about:' + text)
    return [
        get_yandex_gpt_role(),
        {
            "role": "user",
            "text": text
        }
    ]


def create_yandex_gpt_messages_with_reply(reply_text, text):
    print('Asking yaGpt-rest about:' + text)
    return [
        get_yandex_gpt_role(),
        {
            "role": "assistant",
            "text": reply_text
        },
        {
            "role": "user",
            "text": text
        }
    ]


def get_yandex_gpt_role():
    return {
        "role": "system",
        "text": "Ты лучший ассистент со знанием программирования. Говори от женского имени. Говори всегда честно"
    }


# For local testing only
if __name__ == '__main__':
    bot.infinity_polling()
