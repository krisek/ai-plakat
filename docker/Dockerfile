# Use an official Python runtime as a parent image
FROM python:slim-trixie

# ----------------------------------------------------------------------
# Create a non‑root user (UID 1000) and set permissions
# ----------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    # add a user named “appuser” with UID 1000, no password, and a home dir
    useradd -m -u 1000 -s /bin/bash appuser

# Set the working directory in the container to /app and give ownership to appuser
WORKDIR /app
RUN chown appuser:appuser /app

# ----------------------------------------------------------------------
# Install Python dependencies
# ----------------------------------------------------------------------
COPY app/requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Add the application code
ADD app /app
RUN chown -R appuser:appuser /app

# ----------------------------------------------------------------------
# Switch to the non‑root user
# ----------------------------------------------------------------------
USER appuser

# Make port 43264 available to the world outside this container
EXPOSE 43264

# Run the FastAPI app when the container launches
CMD ["uvicorn", "app.py", "--host", "0.0.0.0", "--port", "43264"]