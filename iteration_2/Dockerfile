# check=skip=FromPlatformFlagConstDisallowed
# 1. Base Image: Ubuntu 24.04 (Noble)
FROM --platform=linux/amd64 ubuntu:24.04

# 2. System dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    clang \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 3. Git fix
RUN git config --global --add safe.directory /workspace

# 4. Install Python dependencies globally
RUN pip3 install --break-system-packages \
    fmpy \
    matplotlib \
    pandas \
    tqdm

# 5. Workspace setup
WORKDIR /workspace
CMD ["tail", "-f", "/dev/null"]