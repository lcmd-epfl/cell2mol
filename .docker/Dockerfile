# Container for building the environment
FROM condaforge/mambaforge:4.9.2-5 as conda

RUN python3 -m pip install --no-cache-dir notebook jupyterlab jupyterhub nodejs npm ipywidgets
COPY conda-linux-64.lock .
RUN mamba create --copy -p /env --file conda-linux-64.lock && conda clean -afy
COPY . /pkg

# Distroless for execution
#FROM gcr.io/distroless/base-debian10
#COPY --from=conda /env /env

# User for binder
ARG NB_USER=joyvan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME} /env

# Setup conda env
RUN conda run -p /env python -m pip install --no-deps /pkg
SHELL ["conda", "run", "-p", "/env", "/bin/bash", "-c" ]
RUN python -m ipykernel install --name cell2mol --display-name "cell2mol env"
USER ${NB_USER}


