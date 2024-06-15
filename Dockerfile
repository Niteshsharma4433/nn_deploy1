# Use the official Python base image
FROM python:3.10.14-bookworm

# Upgrade pip and install dependencies
RUN pip install --upgrade pip

# Copy the application source code
COPY src /app/src

# Set the working directory
WORKDIR /app

# Adjust permissions (if needed)
RUN chmod -R 777 /app/src

# Install the Python dependencies
RUN pip install -r /app/src/requirements.txt

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=${PYTHONPATH}:/app/src

# Run the training script
CMD ["python3", "./src/train_pipeline.py"]

# Keep the container running indefinitely
CMD ["tail", "-f", "/dev/null"]
