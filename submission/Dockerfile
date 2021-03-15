# Setup for competition
FROM continuumio/anaconda3:2019.03

RUN apt-get update 
RUN apt-get install -y --no-install-recommends

WORKDIR /app

COPY model .
COPY src .

COPY requirements.txt .

RUN pip install -r requirements.txt 
