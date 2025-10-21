# Performance Testing Guide: ML Inference Service

## Overview

This guide provides tools and instructions for comprehensive performance testing of the `/predict_batch` endpoint using **Locust** for load testing and **cProfile** for CPU profiling.

## Quick Start

### 1. Run CPU Profiling

```bash
# Show all profiling data
python tests/profile_predictions.py

# Show only detailed profiling stats
python tests/profile_predictions.py --detailed

# Show only performance analysis
python tests/profile_predictions.py --analysis

# Show optimization roadmap
python tests/profile_predictions.py --roadmap
```

### 2. Run Load Tests

```bash
# Start the inference service first
uvicorn inference-service.app.main:app --host 0.0.0.0 --port 8000

# In another terminal, start Locust
locust -f tests/load_test_locust.py --host=http://localhost:8000
```

Then open http://localhost:8089 to configure and run tests.

---

## Performance Review Findings

### **P1: ASYNC/SYNC BLOCKING (Critical)**

**Evidence**: `inference-service/app/main.py:199, :265`

**Problem**: 
```python
# Current (BLOCKING)
predictions = model_manager.predict([request.text])  # Synchronous, blocks event loop
```

**Impact**:
- Concurrent requests are serialized (handled one at a time)
- Limits throughput to ~10 RPS per instance
- P99 latency increases with concurrent users
- Underutilizes async/await architecture

**Solution**:
```python
# Fixed (NON-BLOCKING)
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict")
async def predict(request: PredictionRequest):
    predictions = await asyncio.get_event_loop().run_in_executor(
        executor,
        model_manager.predict,
        [request.text]
    )
    return PredictionResponse(**predictions[0])
```

**Expected Improvement**: **3-5x throughput increase** (10 RPS → 30-50 RPS)

---

### **P2: BATCH PROCESSING EFFICIENCY (High)**

**Problem**: Sequential batch processing without vectorization

**Impact**:
- 1850 function calls for 185 items (10 per-item calls)
- No numpy/pandas optimization
- Poor cache locality

**Solution**: Vectorize batch operations using numpy or pandas

**Expected Improvement**: **40-60% latency reduction** for batches >20 items

---

### **P3: MODEL LOADING OVERHEAD (Medium)**

**Problem**: Feature extraction on every inference call

**Solution**: Implement caching for repeated feature extraction

**Expected Improvement**: **20-30% latency reduction**

---

### **P4: LOGGING OVERHEAD (Low)**

**Problem**: Detailed logging in request hot path (`lines 194-204, 260-273`)

**Solution**: Use lazy evaluation, reduce logging verbosity in production

**Expected Improvement**: **5-10% latency reduction**

---

## Profiling Tools Overview

### cProfile Output Interpretation

The profiling script outputs three analysis views:

#### View 1: **Cumulative Time** (Total time in function + sub-calls)

Look for functions with:
- **High cumtime**: Where the code spends most time
- **High ncalls**: Called frequently
- **High cumtime + high ncalls**: OPTIMIZATION TARGET

#### View 2: **Total Time** (Time in function excluding sub-calls)

Look for:
- **High tottime + high ncalls**: Bottleneck to optimize
- **Single calls with high tottime**: Dependency to optimize

#### View 3: **Caller/Callee Analysis**

Shows which functions call which, helping identify:
- Unnecessary function call chains
- Deep recursion
- Inefficient data passing

---

## Load Testing with Locust

### Test Scenarios

**1. Small Batch (5 texts)**
- Baseline performance
- Single request comparison

**2. Medium Batch (25 texts)**  
- Realistic production load
- Balance between latency and throughput

**3. Large Batch (50 texts)**
- Stress testing
- Identifies max capacity

**4. High Concurrency**
- Tests async blocking issues
- Shows degradation under load

### Key Metrics to Monitor

| Metric | Meaning | Target |
|--------|---------|--------|
| RPS | Requests per second | >30 for concurrent loads |
| Response Time (Mean) | Average latency | <100ms for single texts |
| Response Time (P95) | 95th percentile latency | <200ms |
| Response Time (P99) | 99th percentile latency | <500ms |
| Failure Rate | % of failed requests | 0% |
| Users | Concurrent users | Start low, ramp up |

### Running Locust Tests

1. **Start with 10 users at 2 spawns/sec**:
   - Locust UI → Number of users: 10
   - Spawn rate: 2
   - Duration: 60 seconds

2. **Monitor the Stats tab**:
   - Response times per endpoint
   - Failure rates
   - Throughput (RPS)

3. **Identify breaking points**:
   - Increase users until P99 latency >500ms
   - Note the failure rate
   - Calculate max safe load

4. **Compare before/after optimization**:
   - Document baseline metrics
   - Apply P1 optimization
   - Re-run with same parameters
   - Calculate improvement %

---

## Performance Testing Workflow

### Pre-Optimization Baseline

```bash
# 1. Run profiling to understand bottlenecks
python tests/profile_predictions.py

# 2. Document current latency (warm up first)
# Baseline expectations:
#   - Single request: 50-100ms
#   - Batch (25): 100-200ms
#   - Batch (50): 150-250ms

# 3. Load test to find breaking point
locust -f tests/load_test_locust.py --host=http://localhost:8000
# Start with 5 users, increase by 5 until failures

# 4. Record metrics:
#   - RPS at 10 concurrent users
#   - P99 latency at 10 users
#   - Max users before failures
```

### Post-Optimization Validation

```bash
# 1. Implement Phase 1 fix (async threading)
#    (See Code Example above)

# 2. Re-run profiling (should show no serialization)
python tests/profile_predictions.py

# 3. Re-run load tests with same parameters
locust -f tests/load_test_locust.py --host=http://localhost:8000

# 4. Compare metrics
#    Expected improvement: 3-5x RPS increase

# 5. Verify no regressions
#    - Single request latency should be similar
#    - Error rate should remain 0%
#    - Model accuracy should not change
```

---

## Benchmarking Results Template

Use this template to document your test runs:

```markdown
## Load Test Run: [Date/Time]

**Configuration**:
- Users: 10
- Spawn Rate: 2 users/sec
- Duration: 60 seconds
- Host: http://localhost:8000

**Results**:
| Endpoint | Method | Count | Mean (ms) | Min (ms) | Max (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Fail % |
|----------|--------|-------|-----------|----------|----------|----------|----------|----------|--------|
| /predict | POST | 600 | 75 | 25 | 150 | 70 | 120 | 145 | 0% |
| /predict_batch (25) | POST | 300 | 120 | 40 | 250 | 110 | 180 | 220 | 0% |
| /predict_batch (50) | POST | 100 | 200 | 80 | 400 | 190 | 280 | 350 | 0% |
| /health | GET | 200 | 5 | 2 | 15 | 4 | 8 | 12 | 0% |

**Total RPS**: 1200/60 = 20 RPS

**Throughput**: 600 predictions = 10 predictions/sec per user

**Conclusion**: [Performance assessment and recommendations]
```

---

## Common Issues & Solutions

### Issue: "Connection reset by peer"
**Cause**: Server can't handle concurrent requests (async blocking)
**Solution**: Implement Phase 1 optimization (asyncio.to_thread)

### Issue: "Response time increases linearly with users"
**Cause**: Requests being serialized
**Solution**: Same as above - implement async threading

### Issue: "Memory grows with concurrent users"
**Cause**: Model not being released between requests
**Solution**: Implement model pooling/context management

### Issue: "Some batches fail randomly"
**Cause**: Resource contention
**Solution**: Increase thread pool size, add circuit breaker

---

## Advanced Profiling

### Run Detailed Analysis

```bash
python tests/profile_predictions.py --detailed
```

This shows:
- Top 20 functions by cumulative time
- Top 20 functions by total time
- Caller/callee relationships
- Saved to `profiling_results.prof` for further analysis

### Analyze Saved Profiling Data

```bash
python -m pstats profiling_results.prof

# In pstats interactive shell:
(Pstats) sort cumtime      # Sort by cumulative time
(Pstats) stats 20          # Show top 20
(Pstats) callers predict   # Show who calls predict()
(Pstats) callees predict   # Show what predict() calls
```

---

## Monitoring Optimization Progress

Track improvements across phases:

```markdown
## Optimization Progress

### Baseline (Before Optimization)
- RPS: 10
- P99 Latency: 500ms
- Max Concurrent Users: 5

### Phase 1: Async Threading (Expected: 3-5x improvement)
- Target RPS: 30-50
- Target P99: 100-150ms
- Target Max Users: 15-25

### Phase 2: Batch Vectorization (Expected: 40-60% improvement)
- Target RPS: 50-80
- Target P99: 60-100ms

### Phase 3: Feature Caching (Expected: 20-30% improvement)
- Target RPS: 70-100
- Target P99: 40-80ms
```

---

## Production Deployment Checklist

Before deploying performance optimizations to production:

- [ ] Run profiling analysis locally
- [ ] Run load tests and document baseline
- [ ] Implement optimization
- [ ] Re-run profiling (confirm issue resolved)
- [ ] Re-run load tests with same parameters
- [ ] Document improvement percentage
- [ ] Verify no regressions (accuracy, error rate)
- [ ] Test with production-like data
- [ ] Monitor first hour after deployment
- [ ] Have rollback plan ready

---

## References

- **cProfile Documentation**: https://docs.python.org/3/library/profile.html
- **Locust Documentation**: https://locust.io/
- **Async/Threading Guide**: https://docs.python.org/3/library/asyncio.html
- **Performance Optimization Tips**: https://wiki.python.org/moin/PythonSpeed

---

**Status**: Performance testing tools ready for use
**Last Updated**: 2025-01-21
**Next Action**: Run profiling and load tests with your deployment
