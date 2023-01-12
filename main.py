import os
import random
import requests
import telebot
import json
import openai

bot = telebot.TeleBot(os.environ.get('BOT_TOKEN'))
openai.api_key = os.environ.get('OPENAI_API_KEY')


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
    bot.send_message(message.chat.id, "test")


# Command TEST for ci cd
@bot.message_handler(commands=['bot-test-cicd'])
def test(message):
    bot.send_message(message.chat.id, "test cicd")


# Command OPENAI QA
@bot.message_handler(commands=['openai_qa'])
def send_openai_response_to_bot(message):
    # .partition('/openai_qa ')[2] - вырезаем команду от tg из запроса
    bot.reply_to(message, get_openapi_response(message.text.partition('/openai_qa ')[2]))


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


def get_openapi_response(text):
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=text,
        temperature=0,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        #stop=["\n"]
    )

    return response.choices[0].text

# For local testing only
if __name__ == '__main__':
    bot.infinity_polling()
