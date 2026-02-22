# Use Python 3.10 base image
FROM python:3.10-slim

# Avoid root permission issues
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy files
COPY --chown=user . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 10000 for Render
EXPOSE 10000

# Start the app
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]