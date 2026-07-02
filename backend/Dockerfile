# 1. Start with a lightweight Linux box that already has Python 3.12 installed.
FROM python:3.12-slim

# 2. Create a folder inside the container named /app and move inside it.
WORKDIR /app

# 3. Copy app.py from your WSL sandbox into the container's /app folder. 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py .
COPY faiss_index/ faiss_index/
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

