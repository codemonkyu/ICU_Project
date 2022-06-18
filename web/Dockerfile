FROM python:3.8

WORKDIR /usr/src/app

COPY . .

ENV PYTHONIOENCODING=utf-8
ENV ENV_NAME=prod

WORKDIR ./icuweb
RUN pip install -r requirements.txt

CMD python3 manage.py migrate && \
python3 manage.py runserver 0:8000

EXPOSE 8000