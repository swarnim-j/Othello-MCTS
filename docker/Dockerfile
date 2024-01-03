FROM python:3.11

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN python3 train.py

CMD ["python3", "main.py"]