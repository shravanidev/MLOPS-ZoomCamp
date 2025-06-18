FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Copy your Python script into the image
COPY deployment.py .

# Install required libraries
RUN pip install pandas pyarrow scikit-learn==1.5.0 fsspec aiohttp requests

# Default command (optional)
CMD ["python","deployment.py"]
