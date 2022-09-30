import os
import random
import requests
import telebot
import json

bot = telebot.TeleBot(os.environ.get('BOT_TOKEN'))


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

# If message send to private chat
@bot.message_handler(func=lambda message: message.chat.type == "private")
def get_private_message(message):
    bot.send_message(message.chat.id, get_answer(message.text))


# All other message
@bot.message_handler(func=lambda message: True)
def echo(message):
    print(message.text)
    random_int = random.randint(1, 5)
    print(random_int)
    if random_int == 1:
        bot.reply_to(message, get_answer(message.text))


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


# For local testing only
if __name__ == '__main__':
    bot.infinity_polling()
