# Base image with Python 3.9
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    gcc \
    libpq-dev \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements file
COPY ./langgraph/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (adjust as needed)
# COPY ./telegramagent/ .

# Expose the port your API runs on (optional)
# EXPOSE 8000

# Default command to run the app
# Replace with uvicorn or gunicorn as needed
CMD ["python", "main.py"]
