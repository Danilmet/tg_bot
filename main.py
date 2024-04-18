from redis.client import Redis
import telebot
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

BOT_TOKEN = ""

MODEL_NAME = "./models/sbert_plus_multi"
MODEL_TASK = "sentiment-analysis"
REDIS_HOST = "redis"
# REDIS_HOST = "localhost"
SPACY_MODEL = "ru_core_news_sm"
THRESHOLD = 0.9

LABEL_MAP = {
    'LABEL_0': 'greeting',
    'LABEL_1': 'hay',
    'LABEL_2': 'unknown',
}

# Загрузка модели для классификации
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
clf = pipeline(task=MODEL_TASK, model=model, tokenizer=tokenizer, top_k=3)

# Загрузка модели https://spacy.io/models/ru#ru_core_news_sm
ner = spacy.load(SPACY_MODEL)

# Инициализация бота
bot = telebot.TeleBot(BOT_TOKEN)

# Инициализация хранилища Redis
redis_client = Redis(host=REDIS_HOST, port=6379, charset="utf-8", decode_responses=True)


# Функция для распознавания текста с использованием LLM модели
def recognize_text(text):
    result = clf(text)[0]
    print(text, result)
    return LABEL_MAP[result['label']]


def multiple_recognize_text(text):
    result = {}
    for el in clf(text)[0]:
        result[LABEL_MAP[el['label']]] = el['score']
    print(text, result)
    if result['hay'] > THRESHOLD and result['greeting'] > THRESHOLD:
        return 'multi'
    elif result['greeting'] > result['hay'] and result['greeting'] > result['unknown']:
        return 'greeting'
    elif result['hay'] > result['greeting'] and result['hay'] > result['unknown']:
        return 'hay'
    else:
        return 'unknown'


# Распознавание имени с помощью https://spacy.io/models/ru#ru_core_news_sm
def recognize_full_name(text):
    doc = ner(text)
    names = []
    for entry in doc.ents:
        if entry.label_ == "PER":
            names.append(entry.text)
    return names


# Обработчик команды /start
@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет! Я тестовый бот.')


# Обработчик всех текстовых сообщений
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    state = redis_client.get(message.chat.id)
    if state and state == "greeting":
        name = recognize_full_name(message.text)
        if len(name) > 0:
            bot.send_message(message.chat.id, f"Рад познакомиться, {', '.join(name)}")
            redis_client.delete(message.chat.id)
            return
        else:
            bot.reply_to(message, "Как тебя зовут?")
            return
    else:
        recognize_result = multiple_recognize_text(message.text)
        if recognize_result == "multi":
            bot.send_message(message.chat.id, "И тебе привет, отлично!!")
            return
        elif recognize_result == "hay":
            bot.send_message(message.chat.id, "Супер!")
            return
        elif recognize_result == "greeting":
            bot.send_message(message.chat.id, "Привет, как тебя зовут?")
            redis_client.set(message.chat.id, recognize_result)
            return
    bot.reply_to(message, "Я тебя не понял")


# Запуск бота
bot.infinity_polling()
