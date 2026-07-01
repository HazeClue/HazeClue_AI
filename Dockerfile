FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if required for numerical libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Also install fastapi and uvicorn if they aren't in requirements.txt
RUN pip install fastapi uvicorn pydantic

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api.inference_endpoint:app", "--host", "0.0.0.0", "--port", "8000"]
