FROM python:2.7

RUN apt-get update && apt-get install -y nginx
COPY nginx.conf /etc/nginx/
COPY frontend.conf /etc/nginx/conf.d/
COPY ./web.py/templates/ /var/www/frontend/

RUN mkdir /code
WORKDIR /code
COPY ./web.py/requirements.txt /code/
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY ./web.py/ /code/
EXPOSE 8000

CMD service nginx start && python label_debugger.py 8001