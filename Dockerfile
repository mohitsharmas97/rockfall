# Dockerfile (put this in your ROOT landslide/ folder)

# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy requirements file FIRST to leverage Docker cache
COPY requirements.txt .

# 4. Install dependencies
# Using --no-cache-dir keeps the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code
COPY src/backend/ .

# 6. The command to run your app for production
# This uses Gunicorn, a proper production server, and listens on the port Railway provides.
CMD gunicorn --bind 0.0.0.0:$PORT app:app