# Use Python 3.10 slim for compatibility with TensorFlow 2.13
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY . .

# Expose port (Render expects 100% of container to listen on $PORT)
ENV PORT=5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]