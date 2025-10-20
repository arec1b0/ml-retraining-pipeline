"""
Model and Data Monitoring Module

This module leverages Evidently AI to perform crucial MLOps monitoring:
1.  **Data Drift:** Detects statistical changes in the input features
    (e.g., the 'text' data) between a reference dataset and new,
    "current" data.
2.  **Model Performance:** Analyzes the performance (e.g., accuracy, F1)
    of a model on the "current" data and compares it to its performance
    on the "reference" dataset to detect degradation.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.tests import TestAccuracyScore, TestF1Score

from src.utils.logging import get_logger
from src.config.settings import Settings

# Initialize logger for this module
logger = get_logger(__name__)


def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    settings: Settings,
) -> Dict[str, Any]:
    """
    Analyzes data drift and model performance using Evidently AI.

    This function generates an HTML report and returns a summary
    dictionary with key metrics and drift status.

    Args:
        reference_df: The "golden" dataset with features,
                      ground truth ('sentiment'), and predictions.
        current_df: The "production" dataset (or new batch) with
                    the same schema as the reference data.
        settings: The application settings object.

    Returns:
        A dictionary containing analysis results, e.g.:
        {
            "report_path": "path/to/report.html",
            "data_drift_detected": True,
            "model_performance_degraded": False,
            "current_accuracy": 0.85,
            "reference_accuracy": 0.88
        }
    """
    logger.info(
        f"Starting drift analysis. Reference data: {reference_df.shape}, "
        f"Current data: {current_df.shape}"
    )

    # Define column mapping for Evidently
    # 'prediction' is the model's output
    # 'sentiment' is the ground truth (target)
    column_mapping = {
        "target": "sentiment",
        "prediction": "prediction",
        "numerical_features": [],
        "categorical_features": [],
        "text_features": ["text"],
    }

    # 1. Generate Data Drift and Performance Report
    # ------------------------------------------------
    report = Report(
        metrics=[
            DataDriftPreset(text_features=["text"]),
            ClassificationPreset(
                reference_prediction_col="prediction",
                current_prediction_col="prediction"
            ),
        ]
    )

    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    # Ensure the report directory exists
    os.makedirs(settings.EVIDENTLY_REPORTS_PATH, exist_ok=True)

    # Save the HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"drift_report_{timestamp}.html"
    report_path = os.path.join(
        settings.EVIDENTLY_REPORTS_PATH, report_filename
    )
    report.save_html(report_path)
    logger.info(f"Evidently AI report saved to: {report_path}")

    # 2. Extract Key Metrics from the Report
    # ------------------------------------------------
    report_json = report.as_dict()
    
    # Extract data drift status
    # We check if the 'DataDriftPreset' metric indicates dataset-level drift
    try:
        data_drift_metrics = report_json["metrics"][0]["result"]
        data_drift_detected = data_drift_metrics.get("dataset_drift", False)
        logger.info(f"Evidently: Data drift detected: {data_drift_detected}")
    except (IndexError, KeyError) as e:
        logger.warning(f"Could not extract data drift status: {e}")
        data_drift_detected = True  # Default to True to be safe

    # Extract performance metrics
    try:
        classification_metrics = report_json["metrics"][1]["result"]
        current_accuracy = classification_metrics["current"]["accuracy"]
        reference_accuracy = classification_metrics["reference"]["accuracy"]
        logger.info(f"Evidently: Current Accuracy = {current_accuracy:.4f}")
        logger.info(f"Evidently: Reference Accuracy = {reference_accuracy:.4f}")
    except (IndexError, KeyError) as e:
        logger.warning(f"Could not extract performance metrics: {e}")
        current_accuracy = 0.0
        reference_accuracy = 1.0  # Set defaults to trigger retraining

    # 3. Determine if model is degraded
    # ------------------------------------------------
    # Check if performance dropped more than the allowed threshold
    accuracy_drop = reference_accuracy - current_accuracy
    model_performance_degraded = (
        accuracy_drop > settings.MODEL_PERFORMANCE_DEGRADATION_THRESHOLD
    )
    if model_performance_degraded:
        logger.warning(
            f"Model performance degraded! Drop: {accuracy_drop:.4f} "
            f"(Threshold: {settings.MODEL_PERFORMANCE_DEGRADATION_THRESHOLD})"
        )
    else:
        logger.info(f"Model performance is stable. Drop: {accuracy_drop:.4f}")

    # 4. (Optional) Run a Test Suite for specific thresholds
    # ------------------------------------------------
    # This is useful for CI/CD checks
    data_drift_test_suite = TestSuite(
        tests=[
            DataDriftTestPreset(text_features=["text"]),
            TestAccuracyScore(
                gte=reference_accuracy - settings.MODEL_PERFORMANCE_DEGRADATION_THRESHOLD
            ),
            TestF1Score(
                gte=0.6 # Example: set a hard minimum F1 score
            ),
        ]
    )
    data_drift_test_suite.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping
    )
    
    if not data_drift_test_suite.as_dict()["summary"]["all_passed"]:
        logger.warning("Evidently Test Suite failed!")
    else:
        logger.info("Evidently Test Suite passed.")

    return {
        "report_path": report_path,
        "data_drift_detected": data_drift_detected,
        "model_performance_degraded": model_performance_degraded,
        "current_accuracy": current_accuracy,
        "reference_accuracy": reference_accuracy,
    }