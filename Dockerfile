FROM python:3.6
EXPOSE 5000

WORKDIR /qgraph
COPY . .

RUN pip install --no-cache-dir -r web_server/requirements.txt
ENV PYTHONPATH /qgraph

cmd ["python","-u","web_server/server.py"]
