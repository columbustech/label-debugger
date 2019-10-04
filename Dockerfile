FROM python:2.7

RUN apt-get update && apt-get install -y nginx

RUN mkdir /code
WORKDIR /code
COPY ./requirements.txt /code/
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY ./web.py/ /code/
EXPOSE 8000

CMD service nginx start && python label_debugger.py 8000