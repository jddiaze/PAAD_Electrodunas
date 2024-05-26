FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

WORKDIR /
COPY ./app /app
EXPOSE 80

CMD python -m uvicorn app:app --host 0.0.0.0 --port 80