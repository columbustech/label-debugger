FROM ubuntu:16.04
FROM python:3.5

RUN apt-get update && apt-get install -y nginx

RUN mkdir /code
WORKDIR /code
COPY ./requirements.txt /code/
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY ./web.py/ /code/
EXPOSE 8000

CMD python3 label_debugger.py 8000
