# Sentiment Analysis Inference Service

A production-ready FastAPI service for serving sentiment analysis models from MLflow Model Registry. This service is designed as a lightweight, independently deployable microservice optimized for Kubernetes environments.

## Architecture

This inference service is architecturally separate from the ML retraining pipeline:

- **Pipeline (`src/`)**: Batch job for model training and evaluation (heavy dependencies)
- **Inference Service (`inference-service/`)**: Lightweight 24/7 web service (minimal dependencies)

This separation enables:
- Independent deployment and scaling
- Different CI/CD pipelines
- Optimized resource utilization
- Reduced attack surface

## Features

- ✅ **Model Loading**: Loads models from MLflow Model Registry
- ✅ **Single & Batch Predictions**: Efficient inference for 1 or up to 100 texts
- ✅ **Kubernetes-Ready**: Health checks for liveness/readiness probes
- ✅ **Model Metadata**: Track deployed model version and lineage
- ✅ **Production-Grade**: Error handling, logging, validation
- ✅ **Lightweight**: ~150MB Docker image

## API Endpoints

### `GET /health`
Health check endpoint for Kubernetes probes.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "service_name": "sentiment-inference-service",
  "version": "1.0.0"
}
```

### `POST /predict`
Predict sentiment for a single text.

**Request:**
```json
{
  "text": "This product is amazing!"
}
```

**Response:**
```json
{
  "text": "This product is amazing!",
  "sentiment": "positive",
  "confidence": 0.95,
  "model_version": "1"
}
```

### `POST /predict_batch`
Predict sentiment for multiple texts (up to 100).

**Request:**
```json
{
  "texts": [
    "Great product!",
    "Terrible service",
    "Just okay"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "Great product!",
      "sentiment": "positive",
      "confidence": 0.92,
      "model_version": "1"
    },
    {
      "text": "Terrible service",
      "sentiment": "negative",
      "confidence": 0.88,
      "model_version": "1"
    },
    {
      "text": "Just okay",
      "sentiment": "neutral",
      "confidence": 0.65,
      "model_version": "1"
    }
  ]
}
```

### `GET /models/info`
Get metadata about the currently loaded model.

**Response:**
```json
{
  "model_name": "prod-sentiment-classifier",
  "version": "1",
  "run_id": "abc123def456",
  "model_uri": "models:/prod-sentiment-classifier/Production",
  "stage": "Production",
  "loaded_at": "2025-10-20T19:30:00.123456"
}
```

### `GET /`
Root endpoint with service information and links.

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   cd inference-service
   pip install -r requirements.txt
   ```

2. **Set environment variables (optional):**
   ```bash
   export MLFLOW_TRACKING_URI="mlruns"
   export MODEL_URI="models:/prod-sentiment-classifier/Production"
   export LOG_LEVEL="INFO"
   ```

3. **Run the service:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Test the service:**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Single prediction
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This is an amazing product!"}'
   
   # Batch prediction
   curl -X POST http://localhost:8000/predict_batch \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great!", "Terrible!", "Okay."]}'
   
   # Model info
   curl http://localhost:8000/models/info
   ```

5. **View API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker Deployment

1. **Build the Docker image:**
   ```bash
   cd inference-service
   docker build -t sentiment-inference:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 \
     -e MLFLOW_TRACKING_URI="mlruns" \
     -e MODEL_URI="models:/prod-sentiment-classifier/Production" \
     -v $(pwd)/../mlruns:/app/mlruns \
     sentiment-inference:latest
   ```

3. **Test the containerized service:**
   ```bash
   curl http://localhost:8000/health
   ```

### Kubernetes Deployment

1. **Create a deployment manifest (`k8s/deployment.yaml`):**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: sentiment-inference
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: sentiment-inference
     template:
       metadata:
         labels:
           app: sentiment-inference
       spec:
         containers:
         - name: inference
           image: sentiment-inference:latest
           ports:
           - containerPort: 8000
           env:
           - name: MLFLOW_TRACKING_URI
             value: "http://mlflow-service:5000"
           - name: MODEL_URI
             value: "models:/prod-sentiment-classifier/Production"
           resources:
             requests:
               memory: "512Mi"
               cpu: "250m"
             limits:
               memory: "1Gi"
               cpu: "500m"
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 5
             periodSeconds: 5
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: sentiment-inference-service
   spec:
     selector:
       app: sentiment-inference
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8000
     type: LoadBalancer
   ```

2. **Deploy to Kubernetes:**
   ```bash
   kubectl apply -f k8s/deployment.yaml
   ```

3. **Check deployment status:**
   ```bash
   kubectl get pods -l app=sentiment-inference
   kubectl get svc sentiment-inference-service
   ```

## Configuration

Configure the service using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `mlruns` |
| `MODEL_URI` | MLflow Model Registry URI | `models:/prod-sentiment-classifier/Production` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `SERVICE_NAME` | Service identifier | `sentiment-inference-service` |
| `MAX_BATCH_SIZE` | Maximum texts in batch request | `100` |
| `HOST` | Service host binding | `0.0.0.0` |
| `PORT` | Service port | `8000` |

## Model Requirements

The service expects the model to:
- Be registered in MLflow Model Registry with the name `prod-sentiment-classifier`
- Be in the `Production` stage (or specify a different stage/version in `MODEL_URI`)
- Accept a pandas DataFrame with a `text` column
- Return predictions as a list or array

## Performance Considerations

- **Batch Predictions**: Use `/predict_batch` for multiple texts to reduce network overhead
- **Concurrency**: FastAPI handles concurrent requests efficiently with async/await
- **Model Caching**: Model is loaded once on startup and cached in memory
- **Kubernetes Scaling**: Use Horizontal Pod Autoscaler (HPA) for automatic scaling

## Monitoring

The service provides several endpoints for monitoring:

- **Health Checks**: `/health` for Kubernetes liveness/readiness probes
- **Model Metadata**: `/models/info` to verify deployed model version
- **Structured Logs**: JSON logs with request IDs for centralized logging

Integrate with:
- **Prometheus**: Add prometheus-fastapi-instrumentator for metrics
- **Jaeger**: Add OpenTelemetry for distributed tracing
- **Grafana**: Build dashboards for latency, throughput, and error rates

## Security

- ✅ Non-root user in Docker container
- ✅ Input validation with Pydantic
- ✅ CORS middleware (configure for production)
- ✅ No hardcoded credentials
- ⚠️  Add authentication/authorization for production (API keys, OAuth2, etc.)

## Troubleshooting

### Model Not Loading

**Issue**: Service starts but `/health` returns `model_loaded: false`

**Solution**:
1. Check `MLFLOW_TRACKING_URI` points to correct MLflow server
2. Verify model exists: `mlflow models serve -m "models:/prod-sentiment-classifier/Production"`
3. Check service logs: `kubectl logs <pod-name>` or `docker logs <container-id>`

### Prediction Errors

**Issue**: Predictions fail with 500 error

**Solution**:
1. Verify input format matches model expectations
2. Check model was trained with compatible scikit-learn version
3. Review logs for detailed error messages

### High Latency

**Issue**: Slow response times

**Solution**:
1. Use `/predict_batch` for multiple predictions
2. Increase replica count in Kubernetes
3. Profile model inference time
4. Consider model optimization (ONNX, quantization)

## Development

### Project Structure
```
inference-service/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application
│   ├── schemas.py           # Pydantic models
│   ├── model_loader.py      # MLflow model management
│   └── config.py            # Configuration settings
├── Dockerfile               # Production Docker image
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Adding New Features

1. **Add a new endpoint**: Update `app/main.py`
2. **Add request/response models**: Update `app/schemas.py`
3. **Add configuration**: Update `app/config.py`
4. **Rebuild Docker image**: `docker build -t sentiment-inference:latest .`

## License

[Your License Here]

## Support

For issues and questions, please open an issue in the project repository.

