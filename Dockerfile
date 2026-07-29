FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements-dashboard.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dashboard.txt

# Copy app
COPY app.py ./
COPY templates ./templates/
COPY static ./static/
COPY src ./src/
COPY main.py ./

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health').read()"

# Run dashboard
CMD ["python", "app.py"]
