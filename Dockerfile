# Use Python 3.12 slim as the base image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    libatlas-base-dev \
    libglib2.0-0 \
    libcurl4-openssl-dev \
    libssl-dev \
    git \  # Install git to allow fetching from repositories
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the app files into the Docker container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download model file if it doesn't exist (Assuming your app has the download_model function)
RUN python -c "import app; app.download_model()"

# Expose port 8000 to the outside world
EXPOSE 8000

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
