# Use an official Python runtime as a parent image
FROM python:slim-trixie

# Set the working directory in the container to /app
WORKDIR /app

COPY app/requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Add the current directory contents into the container at /app
ADD app /app

# Make port 80 available to the world outside this container
EXPOSE 43264

# Run app.py when the container launches
CMD ["uvicorn", "app.py", "--host", "0.0.0.0", "--port", "43264"]