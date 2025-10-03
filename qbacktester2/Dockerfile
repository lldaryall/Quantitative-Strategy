# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install the project in development mode
RUN pip install --no-cache-dir -e .

# Create directories for data and reports
RUN mkdir -p data reports

# Set default command to show help
CMD ["qbacktester", "--help"]
