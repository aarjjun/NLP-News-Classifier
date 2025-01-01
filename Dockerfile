# Use a lightweight Python base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the application files into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 8050

# Command to run the app when the container starts
CMD ["gunicorn", "dashboard:app", "--bind", "0.0.0.0:8050"]
