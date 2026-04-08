FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

# Start both the HTTP server (background) and keep container alive.
# The evaluator will exec `python inference.py` separately inside the container.
CMD ["python", "-m", "server.app"]
