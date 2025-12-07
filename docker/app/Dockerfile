FROM python:3.11-slim

WORKDIR /usr/src/app

# System deps (optional, but good for many libs)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Default command: run the whole pipeline
CMD ["python", "run_all.py"]
