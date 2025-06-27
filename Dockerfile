# Use a slim Python image
FROM python:3.9-slim

# Set working dir
WORKDIR /app

# Copy your code & requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the default Streamlit port
EXPOSE 3000

# Run Streamlit on the port Vercel will map
CMD ["streamlit", "run", "visualization.py", "--server.port", "3000", "--server.address", "0.0.0.0"]