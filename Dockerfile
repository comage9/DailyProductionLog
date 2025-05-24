FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency definitions
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV DB_PATH="/app/vf.db"
ENV TABLE_NAME="vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)"

# Expose port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 