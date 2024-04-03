FROM ubuntu:22.04 AS builder

ARG PYTHON_VERSION=3.10

WORKDIR /opt

# install basic packages
RUN apt-get update && apt-get -yV upgrade && DEBIAN_FRONTEND=noninteractive apt-get -yV install \
    vim \
    curl wget \
    zip unzip \
    git git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \

# install Python
RUN apt-get update && apt-get -yV upgrade && DEBIAN_FRONTEND=noninteractive apt-get -yV install \
    build-essential libssl-dev libffi-dev \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-distutils python${PYTHON_VERSION}-dev \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install Pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && pip3 --no-cache-dir install --upgrade pip

# install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH $PATH:/root/.local/bin

WORKDIR ${WORKDIR}
CMD ["python"]


FROM builder AS experiment

ENV WORKDIR /workspaces/edinburgh_digest/
WORKDIR ${WORKDIR}

RUN poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project false

# install Python packages
COPY poetry.lock pyproject.toml ${WORKDIR}
RUN poetry install --no-root


# add source directory to pythonpath
COPY src ${WORKDIR}/src
RUN poetry install
COPY . ${WORKDIR}


# cleanup
RUN apt-get autoremove -y && apt-get clean && \
    rm -rf /usr/local/src/* && rm -rf ~/.cache/pip && poetry cache clear pypi --all