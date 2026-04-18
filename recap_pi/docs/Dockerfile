FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    lmodern \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/recap_pi/docs

CMD ["bash"]
