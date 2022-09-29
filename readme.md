## Локальный запуск

Для локального запуска и тестирования надо:
 - Создать своего тг-бота
 - Прописать его токен в переменные окружения
 - В файле main.py запустить программу с использованием пуллинга сообщений из тг


#### Создание тг-бота:

- Находим бота `@BotFather` и вызываем у него команду `/newbot`
- Задаем имя для отображения и "username" (должен быть с постфиксом bot)
- Копируем токен, который имеет вид: `5768543123:AAAAAAAAAAAAA_AA`


#### Добавить токен в переменные окружения
 // todo

## Зависимости:
`pip install pyTelegramBotAPI`