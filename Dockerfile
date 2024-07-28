# Build stage
FROM nvcr.io/nvidia/cuda:12.5.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  git \
  && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install hf_transfer and wheel
RUN pip3 install --no-cache-dir --upgrade pip && \
  pip3 install -U hf_transfer wheel

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -U flash-attn

# Runtime stage
FROM nvcr.io/nvidia/cuda:12.5.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set default values for customisable parameters
ENV SYSTEM_MESSAGE="Carefully follow the users request."
ENV TEMPERATURE=0.3
ENV TOP_P=0.7
ENV MAX_NEW_TOKENS=2048
ENV MAX_INPUT_TILES=12
ENV REPETITION_PENALTY=1.0
ENV MODEL_NAME=OpenGVLab/InternVL2-8B
ENV LOAD_IN_8BIT=1

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  supervisor \
  git \
  vim \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set up working directory
WORKDIR /app

# Copy application files
COPY . /app
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create a volume for HuggingFace cache
VOLUME /root/.cache/huggingface

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose ports
# 21001 = controller
# 21002 = worker
# 7860 = gradio web interface
EXPOSE 7860 21001 21002

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]
