FROM python:3.11-slim

WORKDIR /app

# Pillow needs a couple of runtime libs for JPEG/PNG handling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 --no-cache-dir install -r requirements.txt

COPY . ./

ENTRYPOINT [ "python3", "-u", "transform.py" ]
