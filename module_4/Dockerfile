FROM python:3.10.13-slim

WORKDIR /app
COPY [ "model_from_zoomcamp.bin", "model.bin" ]

RUN pip install -U pip & pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "result.py", "./result.py" ]
RUN mkdir "output"

CMD [ "python", "-u", "result.py", "2023", "5" ]
