FROM continuumio/miniconda3:24.9.2-0

ARG SAM2_REPOSITORY=https://github.com/facebookresearch/sam2.git
ARG SAM2_COMMIT=2b90b9f5ceec907a1c18123530e92e794ad901a4

WORKDIR /workspace/occlusion-mask-generation

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml requirements.txt ./
RUN conda env create --file environment.yml

ENV CONDA_DEFAULT_ENV=OMG
ENV PATH=/opt/conda/envs/OMG/bin:$PATH

RUN python -m pip install --upgrade -r requirements.txt

COPY . .

RUN if [ ! -d external/sam2/.git ]; then git clone "${SAM2_REPOSITORY}" external/sam2; fi \
    && git -C external/sam2 checkout "${SAM2_COMMIT}" \
    && SAM2_BUILD_CUDA=0 python -m pip install --no-deps --no-build-isolation -e external/sam2

CMD ["/bin/bash"]
