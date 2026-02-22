FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]