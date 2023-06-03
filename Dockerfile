# Copy code
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Set the environment variable for the Flask application
ENV FLASK_APP=chatbot.py

# Run the Flask application when the container launches
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080"]
