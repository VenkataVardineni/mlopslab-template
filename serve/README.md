# Model Serving

This directory contains the FastAPI model serving application.

## Quick Start

### Local Development

```bash
# Ensure model artifacts exist (run pipeline first)
make repro

# Run the server
cd serve
uvicorn app:app --reload

# Or use Python directly
python serve/app.py
```

The API will be available at http://localhost:8000

### Docker

```bash
# Build the image
docker build -t mlopslab-serve -f serve/Dockerfile .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  mlopslab-serve
```

## API Endpoints

### GET /

Root endpoint with API information.

### GET /health

Health check endpoint.

### POST /predict

Generate forecast from a context window.

**Request Body:**
```json
{
  "context": [1.0, 2.0, 3.0, ...],  // At least context_length values
  "timestamp": "2024-01-01"  // Optional
}
```

**Response:**
```json
{
  "forecast": [4.5, 5.2, ...],      // Median forecast (p50)
  "forecast_p10": [3.8, 4.5, ...],  // Lower bound (p10)
  "forecast_p90": [5.2, 6.0, ...],  // Upper bound (p90)
  "horizon": 7
}
```

## Example Usage

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "context": [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5]
  }'

# Using Python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "context": [8.0, 8.5, 9.0, ...]  # 30+ values
    }
)
print(response.json())
```

## Requirements

- Model artifacts must be present in `artifacts/model/`
- Scaler must be present in `artifacts/preprocess/scaler.json`
- Context window must have at least `context_length` values (default: 30)

