# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Make port 5000 available outside the container
EXPOSE 5000

# Set the command to run when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]