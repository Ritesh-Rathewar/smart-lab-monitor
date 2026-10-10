# Start from a small computer that already has Python 3.12
FROM python:3.12-slim

# Work inside a folder called /app
WORKDIR /app

# Copy the shopping list first and install the libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of our project
COPY . .

# Train the model inside the box so it matches this Python
RUN python train_model.py

# The app listens on port 5000
EXPOSE 5000

# Start the app with gunicorn (a professional web server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "app:app"]