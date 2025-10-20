# Automated Model Retraining Pipeline 🔄📊

This project implements a complete, end-to-end MLOps pipeline for a sentiment analysis model. It is designed to complement the `KubeSentiment` project by focusing on the full MLOps feedback loop: data validation, model monitoring (drift detection), automated retraining, and model registration.

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

---

## 🚀 Getting Started (Windows 11)

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
````

### 2\. Initialize MLOps Tooling

These commands only need to be run once to set up the project.

```bash
# Initialize Great Expectations
# This scaffolds the 'great_expectations/' directory
great_expectations init
```

### 3\. Initialize DVC (Simulated)

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

### 4\. Run the Retraining Pipeline

The pipeline is defined and managed by Prefect.

```bash
# 1. Start the Prefect server (in a separate terminal)
# This provides the UI and API for managing flows.
prefect server start

# 2. Run the main retraining flow
# This will execute the entire pipeline: validation, training, evaluation,
# drift detection, and registration.
python src/pipeline/flows.py

# 3. Open the Prefect UI in your browser to see the run:
# [http://127.0.0.1:4200](http://127.0.0.1:4200)
```

### 5\. View Results

  * **Prefect UI:** [http://127.0.0.1:4200](https://www.google.com/url?sa=E&source=gmail&q=http://127.0.0.1:4200)
      * Observe the flow run graph and task logs.
  * **MLflow UI:** (In a new terminal)
    ```bash
    mlflow ui
    ```
      * Open [http://127.0.0.1:5000](https://www.google.com/search?q=http://127.0.0.1:5000) to view experiments, compare runs, and see registered models.
  * **Great Expectations Data Docs:**
      * Open `great_expectations/uncommitted/data_docs/local_site/index.html` to see data validation results.
  * **Evidently AI Reports:**
      * Open the HTML files generated in `reports/evidently/` to see drift analysis.

-----

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
