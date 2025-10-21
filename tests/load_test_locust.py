"""
Load Testing Script for /predict_batch Endpoint
Uses Locust to simulate concurrent users and measure performance
"""

from locust import HttpUser, task, between
import random


class PredictionLoadTest(HttpUser):
    """
    Simulates users making prediction requests to the inference service
    """
    
    wait_time = between(0.5, 2.0)  # Wait 0.5-2 seconds between requests
    
    def on_start(self):
        """Initialize test data once per user"""
        self.sample_texts = [
            "This product is amazing! Highly recommend.",
            "Terrible service, will not return.",
            "It's okay, nothing special.",
            "Love it! Best purchase ever.",
            "Waste of money, very disappointed.",
            "Great quality and fast shipping.",
            "Not what I expected at all.",
            "Perfect! Exceeded my expectations.",
            "Average product, average price.",
            "Fantastic experience from start to finish.",
        ]
    
    @task(1)
    def predict_single_batch_small(self):
        """Small batch: 5 texts"""
        batch = random.sample(self.sample_texts, min(5, len(self.sample_texts)))
        self.client.post(
            "/predict_batch",
            json={"texts": batch},
            name="predict_batch_small"
        )
    
    @task(2)
    def predict_medium_batch(self):
        """Medium batch: 25 texts"""
        batch = [random.choice(self.sample_texts) for _ in range(25)]
        self.client.post(
            "/predict_batch",
            json={"texts": batch},
            name="predict_batch_medium"
        )
    
    @task(1)
    def predict_large_batch(self):
        """Large batch: 50 texts"""
        batch = [random.choice(self.sample_texts) for _ in range(50)]
        self.client.post(
            "/predict_batch",
            json={"texts": batch},
            name="predict_batch_large"
        )
    
    @task(3)
    def predict_single(self):
        """Single prediction for comparison"""
        text = random.choice(self.sample_texts)
        self.client.post(
            "/predict",
            json={"text": text},
            name="predict_single"
        )
    
    @task(1)
    def health_check(self):
        """Health check - should be fast"""
        self.client.get("/health", name="health")


class HighConcurrencyUser(HttpUser):
    """
    Simulates high-concurrency scenario with minimal wait time
    """
    
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        """Initialize with batch data"""
        self.batch_data = [
            "Text " + str(i) for i in range(100)
        ]
    
    @task
    def rapid_batch_predictions(self):
        """Make rapid batch predictions"""
        batch = random.sample(self.batch_data, 20)
        self.client.post(
            "/predict_batch",
            json={"texts": batch},
            name="rapid_batch"
        )


if __name__ == "__main__":
    print("""
    Load Testing Script for ML Inference Service
    
    Usage:
        locust -f tests/load_test_locust.py --host=http://localhost:8000
    
    Then open: http://localhost:8089
    
    Test Scenarios:
    1. Small batches (5 texts) - baseline
    2. Medium batches (25 texts) - realistic load
    3. Large batches (50 texts) - stress test
    4. Single predictions - comparison
    5. High concurrency - identify bottlenecks
    """)
