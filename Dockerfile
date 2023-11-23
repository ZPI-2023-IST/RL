FROM python:3.10-slim-buster

EXPOSE 5000

WORKDIR /app

COPY . .

RUN python3 -m pip install .

ENTRYPOINT ["python3"]
CMD ["rl/api/main.py"]
