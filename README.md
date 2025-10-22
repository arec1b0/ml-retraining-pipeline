# Automated Model Retraining Pipeline 🔄📊

## Purpose
This project implements a complete, end-to-end MLOps pipeline for a sentiment analysis model. It is designed to complement the KubeSentiment project by focusing on the full MLOps feedback loop: data validation, model monitoring (drift detection), automated retraining, and model registration.

The core of this system is a **Prefect** workflow that is triggered when model performance or data quality degrades, ensuring the production model remains accurate and robust over time.

## 🏛️ Core Architecture & Technology

This system is built using a modern, Python-native MLOps stack:

* **Workflow Orchestration:** **Prefect**
    * Used to define, schedule, and execute the entire retraining pipeline, from data validation to model registration.
* **Experiment Tracking:** **MLflow**
    * Logs all training runs, parameters, metrics, and model artifacts.
    * Manages the model lifecycle via the MLflow Model Registry (e.g., staging, production).
* **Data Validation:** **Great Expectations (GE)**
    * Defines "Expectation Suites" to validate the schema, integrity, and statistical properties of incoming data before training.
* **Model/Data Monitoring:** **Evidently AI**
    * Continuously monitors for data drift, concept drift, and model performance degradation by comparing production data against a stable reference dataset.
* **Data Version Control:** **DVC**
    * Versions large data files (like `feedback.csv`) and models, storing them in remote storage (e.g., S3, MinIO) while keeping lightweight pointers in Git.
* **Containerization:** **Docker**
    * Packages the entire application, including all dependencies and code, for consistent execution by Prefect agents.
* **CI/CD Integration:** **GitHub Actions**
    * Automated pipeline linking: When a model is promoted to Production, the CD pipeline is automatically triggered via GitHub Actions API.

---

## 🚀 Getting Started and Usage Guide

This guide provides a comprehensive overview of how to set up, run, and interpret the results of the automated retraining pipeline.

### Prerequisites

* [Python 3.10+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/downloads)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Ensure it's running in Linux container mode)
* [DVC](https://dvc.org/doc/install) (`pip install dvc[s3]`)

### 1. Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/arec1b0/AutomatedModelRetrainingPipeline.git
cd AutomatedModelRetrainingPipeline

# Create a virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Use `.venv\Scripts\activate` in CMD/PowerShell

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create your local environment file
# (Copy .env.example and leave default values for local setup)
copy .env.example .env
```

### 2. Initialize MLOps Tooling

These commands only need to be run once to set up the project.

```bash
# Initialize Great Expectations
# This scaffolds the 'great_expectations/' directory
great_expectations init
```

### 3. Initialize DVC (Simulated)

This project is configured to use DVC with an S3-compatible backend. For local development, you would typically use a local remote or a Docker-based MinIO server.

```bash
# (Optional) Initialize DVC if you were starting from scratch
# dvc init
# dvc remote add -d local_storage s3://your-s3-bucket/dvc-cache -f
# git commit -m "Configure DVC remote"

# Pull the data from the (simulated) remote storage
# This will download the .csv files defined in the data/*.dvc files (once we create them)
dvc pull
```

### 4. Running the Retraining Pipeline

The pipeline is defined and managed by Prefect. The main entrypoint is `src/pipeline/flows.py`, which orchestrates all the steps.

```bash
# 1. Start the Prefect server (in a separate terminal)
# This provides the UI and API for managing flows.
prefect server start

# 2. Run the main retraining flow
# This will execute the entire pipeline: validation, training, evaluation,
# drift detection, and registration.
# The `force_retrain=True` flag is necessary for the first run to ensure
# a baseline model is trained and registered.
python src/pipeline/flows.py

# 3. Open the Prefect UI in your browser to see the run:
# [http://127.0.0.1:4200](http://127.0.0.1:4200)
```

### 5. Interpreting the Outputs and Viewing Results

After a successful run, you can inspect the outputs from the various MLOps tools:

*   **Prefect UI:** [http://127.0.0.1:4200](http://127.0.0.1:4200)
    *   Observe the flow run graph, view detailed logs for each task, and see the final status of the pipeline.

*   **MLflow UI:**
    *   Launch the MLflow UI from your terminal:
        ```bash
        mlflow ui
        ```
    *   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to:
        *   **Compare Runs:** Analyze the parameters and metrics from different training runs.
        *   **View Artifacts:** Inspect the model files and other artifacts logged during training.
        *   **Manage Models:** Check the Model Registry to see which models are in "Staging" or "Production".

*   **Great Expectations Data Docs:**
    *   Open `great_expectations/uncommitted/data_docs/local_site/index.html` in your browser.
    *   Review the data validation results to see if the raw data met the quality expectations defined in the suite.

*   **Evidently AI Reports:**
    *   Navigate to the `reports/evidently/` directory.
    *   Open the most recent HTML report to visualize data drift and model performance degradation metrics. This report is critical for understanding *why* a model retraining was triggered.

---
## 🔗 CT-CD Integration (Continuous Training → Continuous Deployment)

This pipeline features **automated CT-CD linkage** that closes the loop between model training and deployment:

**What it does:**
- 🤖 When a new model is promoted to Production in MLflow, it automatically triggers the CD pipeline
- 🚀 The CD pipeline builds and pushes a new Docker image for the inference service
- 📊 Full traceability: Model version and accuracy metadata are passed to the deployment

**Quick Setup:**
```bash
# 1. Set environment variables in .env
ENABLE_CD_TRIGGER=true
GITHUB_TOKEN=your_github_token
GITHUB_REPO_OWNER=your-username
GITHUB_REPO_NAME=ml-retraining-pipeline

# 2. That's it! The next time a model is promoted, CD will trigger automatically
```

**Documentation:**
- 📖 [Quick Start Guide](docs/QUICK_START_CT_CD.md) - 5-minute setup
- 📖 [Full CT-CD Documentation](docs/CT_CD_INTEGRATION.md) - Architecture & best practices
---
## 📂 Project Structure

```
AutomatedModelRetrainingPipeline/
├── .dvc/                   # DVC metadata
├── .github/                # GitHub Actions (CI/CD)
├── data/                   # Data versioned by DVC (raw, processed, reference)
├── great_expectations/     # Great Expectations suites, checkpoints, and docs
├── mlruns/                 # Local MLflow experiment tracking data
├── reports/
│   └── evidently/          # HTML drift reports
├── src/                    # Main Python source code
│   ├── config/             # Pydantic settings management
│   ├── data_validation/    # GE helper functions
│   ├── model_monitoring/   # Evidently helper functions
│   ├── pipeline/           # Prefect flows and tasks
│   │   ├── flows.py        # Main retraining flow
│   │   └── tasks/          # Atomic tasks (train, evaluate, etc.)
│   └── utils/              # Utility functions (e.g., logging)
├── terraform/              # Infrastructure as Code (e.g., S3, MLflow server)
├── Dockerfile              # Containerizes the application
├── prefect.yaml            # Prefect deployment definitions
└── requirements.txt        # Python dependencies
```
