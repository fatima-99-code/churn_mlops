# Backup/Notes section
# RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

# Local virtual environment copy karein
COPY venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY app.py schemas.py ./
COPY mlruns ./mlruns

EXPOSE 8000

CMD ["/app/venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]