FROM python:3.11-slim

WORKDIR /app

# (Optional but helpful) avoid generating .pyc, ensure logs show up
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code + model artifact
COPY src ./src
COPY app ./app
COPY artifacts ./artifacts

# Streamlit runs on 8501 by default
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
