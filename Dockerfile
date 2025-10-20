# Use an official Python runtime as a parent image
# Using slim-buster for a smaller image size
FROM python:3.11-slim-buster

# Set environment variables
# -------------------------
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files.
# PYTHONUNBUFFERED: Ensures Python output (e.g., print statements) is sent
#                   straight to the terminal without buffering.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies
# ---------------------------------
# git: Required for DVC to pull data if it's stored in a git-tracked location
#      (though we use S3, it's good practice).
# curl: Useful for health checks or downloading files.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# ---------------------------
# First, copy only the requirements.txt file to leverage Docker layer caching.
# This layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire project source code
# -----------------------------------
# This copies the 'src' directory, 'prefect.yaml', 'great_expectations/'
# and other necessary files into the container's working directory.
COPY . .

# Initialize Great Expectations (non-interactively)
# -------------------------------------------------
# This ensures the 'great_expectations' directory is properly configured
# within the container environment.
RUN great_expectations init --ci

# Expose ports
# ------------
# Expose the default Prefect UI/API port (if we were to serve it from here)
EXPOSE 4200

# Default command
# ----------------
# This container is intended to be run with a Prefect agent or to execute flows.
# We'll set a default command that can be easily overridden.
CMD ["prefect", "version"]