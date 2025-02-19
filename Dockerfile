# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV FLASK_APP=chatbot.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Run chatbot.py when the container launches

# CMD gunicorn -w 4 -b 0.0.0.0:8000 chatbot:app
CMD gunicorn -w 4 -b :$PORT chatbot:app
# CMD gunicorn --bind :$PORT chatbot:app
# CMD ["flask", "run", "--port", "80"]
