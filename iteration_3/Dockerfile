# check=skip=FromPlatformFlagConstDisallowed
FROM --platform=linux/amd64 ubuntu:24.04

# System dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    clang \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Git fix
RUN git config --global --add safe.directory /workspace

# Install Python dependencies globally
RUN pip3 install --break-system-packages \
    fmpy \
    matplotlib \
    pandas \
    tqdm

# Workspace setup
WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]