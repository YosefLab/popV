FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
FROM python:3.11 AS base

RUN pip install --no-cache-dir uv

RUN uv pip install --system --no-cache torch torchvision torchaudio "jax[cuda12]"
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com cuml-cu12 cudf-cu12 cugraph-cu12

CMD ["/bin/bash"]

FROM base AS build

ENV POPV_PATH="/usr/local/lib/python3.11/site-packages/popv"

COPY . ${POPV_PATH}

ARG DEPENDENCIES=""
RUN uv pip install --system "popv[${DEPENDENCIES}] @ ${POPV_PATH}"
