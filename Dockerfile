FROM python:3.12-slim
WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

COPY . .

ENV PORT=5000
EXPOSE 5000

CMD ["python", "application.py"]
