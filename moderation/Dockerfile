# Minimal Python Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
# Create a dummy requirement if missing so pip doesn't fail
RUN touch requirements.txt && pip install -r requirements.txt
CMD ["python", "app/api.py"]