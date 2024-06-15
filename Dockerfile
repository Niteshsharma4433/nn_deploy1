# Use the official Python base image
FROM python:3.10.14-bookworm

# Upgrade pip and install dependencies
RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy the application source code
COPY src /app/src

# Adjust permissions
RUN chmod -R 777 /app/src

# Install the Python dependencies
RUN pip install -r /app/src/requirements.txt

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=${PYTHONPATH}:/app/src

# Create entrypoint script
RUN echo '#!/bin/bash\n' \
    'if [ "$1" == "train" ]; then\n' \
    '  python3 /app/src/train_pipeline.py\n' \
    'elif [ "$1" == "predict" ]; then\n' \
    '  python3 /app/src/predict.py\n' \
    'else\n' \
    '  echo "Usage: $0 {train|predict}"\n' \
    '  exit 1\n' \
    'fi\n' > /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint to the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command to keep the container running
CMD ["tail", "-f", "/dev/null"]
