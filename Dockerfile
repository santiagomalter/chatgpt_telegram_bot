FROM python:3.8-slim

ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random
ENV PYTHONDONTWRITEBYTECODE 1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100
ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git

RUN apt-get update
RUN apt-get install -y python3 python3-pip python-dev build-essential python3-venv ffmpeg git

# Copy SSH key for git private repos
# Create .ssh directory
RUN mkdir -p /root/.ssh
RUN chmod 700 /root/.ssh

# Disable strict host key checking
RUN echo "Host *\n\tStrictHostKeyChecking no\n" > /root/.ssh/config
RUN chmod 600 /root/.ssh/config


RUN mkdir -p /code
ADD . /code
WORKDIR /code

RUN pip3 install -r requirements.txt

CMD ["bash"]
