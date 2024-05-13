# For more information, please refer to https://aka.ms/vscode-docker-python
#FROM python:3.10
FROM nvidia/cuda:11.7.0-base-ubuntu22.04

RUN apt-get -y update 
RUN apt-get -y install software-properties-common 
RUN add-apt-repository ppa:deadsnakes/ppa 
RUN apt install -y python3.10

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .

# Install pip requirements
RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get install -y python3-pip
RUN pip install --upgrade pip
RUN pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install tqdm

CMD ["python3", "main.py"]
