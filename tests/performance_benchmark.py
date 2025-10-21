"""
Performance Benchmark Script
Measures inference service performance using Apache Locust
Captures RPS, latency, and error metrics
"""

import subprocess
import time
from pathlib import Path


def run_locust_benchmark(duration_seconds=30, users=10, spawn_rate=2):
    """
    Run Locust load test in headless mode and capture metrics.

    Args:
        duration_seconds: How long to run the test
        users: Number of concurrent users
        spawn_rate: How many users to spawn per second

    Returns:
        Dictionary with performance metrics
    """
    print(f"\n{'='*70}")
    print(
        f"LOAD TEST: {users} users, spawn_rate {spawn_rate}, "
        f"duration {duration_seconds}s"
    )
    print(f"{'='*70}\n")

    try:
        # Run locust in headless mode
        cmd = [
            "locust",
            "-f",
            "tests/load_test_locust.py",
            "--host",
            "http://localhost:8000",
            f"--users={users}",
            f"--spawn-rate={spawn_rate}",
            f"--run-time={duration_seconds}s",
            "--headless",
            "--csv=.cursor/performance_test",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=duration_seconds + 30
        )

        print("LOCUST OUTPUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Parse CSV results if they exist
        stats_file = Path(".cursor/performance_test_stats.csv")
        if stats_file.exists():
            print("\n" + "=" * 70)
            print("PERFORMANCE METRICS")
            print("=" * 70)
            with open(stats_file, "r") as f:
                content = f.read()
                print(content)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("WARNING: Locust test timed out")
        return False
    except Exception as e:
        print(f"ERROR: Failed to run Locust: {e}")
        return False


def main():
    """Run performance benchmarks"""
    print("\n" + "=" * 70)
    print("ML INFERENCE SERVICE PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Check if service is running
    print("\nChecking service health...")
    try:
        import requests  # type: ignore[import-untyped]

        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ Service is healthy and ready for testing")
        else:
            print("✗ Service returned unexpected status code")
            return
    except Exception as e:
        print(f"✗ Cannot connect to service: {e}")
        print("Please start the inference service first:")
        print("  cd inference-service && python -m uvicorn app.main:app")
        return

    # Create output directory
    Path(".cursor").mkdir(exist_ok=True)

    # Run tests with different concurrency levels
    test_configs = [
        {"users": 5, "spawn_rate": 1, "duration": 30},
        {"users": 10, "spawn_rate": 2, "duration": 30},
    ]

    results = []
    for config in test_configs:
        success = run_locust_benchmark(
            duration_seconds=config["duration"],
            users=config["users"],
            spawn_rate=config["spawn_rate"],
        )
        results.append((config, success))
        time.sleep(2)  # Cool down between tests

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    for config, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(
            f"{status}: {config['users']} users, "
            f"spawn_rate {config['spawn_rate']}"
        )

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Compare metrics before and after optimization")
    print("2. Look for improvement in RPS (requests per second)")
    print("3. Check P99 latency reduction")
    print("4. Verify error rate is 0%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
