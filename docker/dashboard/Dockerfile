FROM python:3.11-slim

WORKDIR /usr/src/app

# Install Python deps for the dashboard
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    numpy \
    matplotlib \
    minio

# Copy app code
COPY app.py .

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
