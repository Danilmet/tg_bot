FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download ru_core_news_sm

COPY . .

CMD [ "python", "./main.py" ]