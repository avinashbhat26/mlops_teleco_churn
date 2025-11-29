Telco Churn Prediction – Complete End-to-End MLOps Pipeline (Local + Open-Source Tools)

This project builds a complete end-to-end MLOps system using ONLY open-source, fully local tools.
It covers the entire lifecycle of an ML project:

✔ Data ingestion
✔ Exploratory data analysis
✔ Preprocessing
✔ Feature engineering
✔ Model training
✔ Evaluation
✔ Experiment tracking
✔ Pipeline orchestration
✔ Deployment (FastAPI)
✔ Containerization (Docker)
✔ Monitoring (Evidently AI)
✔ CI/CD Simulation (GitHub Actions)
✔ Version control (Git & GitHub)
✔ Data & model versioning (DVC)

This project was developed by
Avinash P Bhat,Nikhil Bharadwaj, Ganesh Prasad D M and Nandish,Executive M.Tech Applied AI
Visvesvaraya National Institute of Technology (VNIT) Nagpur

🧠 1. Project Objective

Build a fully local end-to-end MLOps pipeline using open-source tools for:

Automating ML workflows

Versioning of data & models via DVC

Tracking experiments via MLflow

Deploying model using FastAPI

Monitoring live data drift using Evidently

Simulating CI using GitHub Actions

Dataset chosen: Kaggle – Telco Customer Churn (≈ 7000 rows)

📁 2. Project Structure
mlops_teleco_churn/
│
├── app/                        # FastAPI application
│   └── main.py
│
├── src/                        # All MLOps pipeline scripts
│   ├── data_ingestion.py       # Load CSV, train-test split
│   ├── preprocess.py           # Encoding, scaling, save preprocessor
│   ├── train.py                # Train model + log to MLflow
│   ├── evaluate.py             # Evaluate model + log metrics
│   ├── monitoring.py           # Generate Evidently drift report
│   ├── prefect_flow.py         # Prefect orchestration pipeline
│   └── predict.py              # Helper for FastAPI prediction
│
├── data/                       
│   ├── raw/                    # Raw dataset (DVC-tracked)
│   └── processed/              # train.csv / test.csv
│
├── models/                     # Model + preprocessor
│
├── reports/
│   └── evidently_report.html   # Drift monitoring report
│
├── .dvc/                       # DVC metadata
├── .github/workflows/ci.yml    # CI Simulation Workflow
│
├── dvc.yaml                    # DVC pipeline
├── dvc.lock
├── Dockerfile
├── params.yaml                 # Central config file
├── requirements.txt
└── README.md

⚙️ 3. Tools Used
ML Pipeline

Python

Pandas, NumPy, Scikit-learn

MLOps Tools
Tool	Purpose
Git & GitHub	Version control
DVC	Dataset & pipeline versioning
MLflow	Experiment & model tracking
Prefect	Pipeline orchestration
FastAPI	Model deployment
Docker	Containerization
Evidently AI	Batch monitoring
GitHub Actions	CI simulation
🔧 4. How to Set Up the Project (VERY IMPORTANT)


🛠️ Step 1 — Clone the repository
git clone https://github.com/avinashbhat26/mlops_teleco_churn.git
cd mlops_teleco_churn

🛠️ Step 2 — Create Virtual Environment
python -m venv .venv


Activate it:

Windows:
.\.venv\Scripts\activate

🛠️ Step 3 — Install all dependencies
pip install -r requirements.txt


This installs:

sklearn

pandas

numpy

mlflow

fastapi

uvicorn

prefect

evidently

dvc

Everything needed is included.

📦 5. Running the Complete MLOps Pipeline

All pipeline steps are automated with DVC.

Run the entire pipeline:
dvc repro


This runs:

1️⃣ src/data_ingestion.py
✔ Loads dataset
✔ Performs train-test split
✔ Saves processed data
✔ Tracks dataset via DVC

2️⃣ src/preprocess.py
✔ Label encode / one-hot encode
✔ Standard scaling
✔ Save preprocessor

3️⃣ src/train.py
✔ Train model
✔ Log parameters + metrics to MLflow
✔ Save model.pkl

4️⃣ src/evaluate.py
✔ Compute accuracy, precision, recall, f1
✔ Log evaluations to MLflow

Outputs stored in:

data/processed/
models/
mlruns/

📊 6. Using MLflow for Experiment Tracking

Start MLflow UI:

mlflow ui


Open browser:
👉 http://127.0.0.1:5000

You will see:

Experiment name: telco_churn_experiment

run metrics

parameters

confusion matrix

model artifact downloads

🌐 7. Running the FastAPI Deployment

Start API:

uvicorn app.main:app --reload


Visit:
👉 http://127.0.0.1:8000/docs

You get full Swagger UI.

Example prediction input:
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "InternetService": "Fiber optic",
  "Contract": "Month-to-month",
  "MonthlyCharges": 70.5,
  "TotalCharges": 845.3
}

Output:
{
  "prediction": 0,
  "churn_probability": 0.29
}

📉 8. Monitoring with Evidently (Data Drift Report)

Run:

python -m src.monitoring


This generates:

reports/evidently_report.html


Teacher can open this file in browser to see:

Feature drift

Data stability

Target drift

Correlations

Histograms

⚙️ 9. Orchestrating Pipeline with Prefect

Start pipeline:

python -m src.prefect_flow


This executes:

ingestion → preprocess → train → evaluate

Retries on failure

Logs to Prefect engine

🐳 10. Running with Docker
Build Docker image:
docker build -t telco-app .

Run container:
docker run -p 8000:8000 telco-app


API available at:

👉 http://127.0.0.1:8000/docs

🔄 11. CI/CD Simulation (GitHub Actions)

Workflow file at:

.github/workflows/ci.yml


Runs automatically on each push:

Install dependencies

Validate repo structure

Dummy tests

DVC + MLflow checks

Linting

This simulates a real CI pipeline.

🧪 12. Testing the Prediction Node (Local or Docker)

Use predict endpoint:

curl -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d "{\"gender\":\"Female\",\"SeniorCitizen\":0,...}"

✔️ 13. Deliverables (All Included)

Your project satisfies 100% of assignment:

✔ GitHub Repo

✔ Data Versioning (DVC)

✔ MLOps Pipeline

✔ MLflow Tracking

✔ Docker Deployment

✔ FastAPI Serving

✔ Prefect Flow

✔ Evidently Monitoring

✔ CI/CD Workflow

✔ Detailed README (This file)

✔ Demo script (available upon request)

👨‍🎓 14. Authors

Avinash P Bhat
Nikhil Bharadwaj
Ganesh Prasad DM
Nandish

M.Tech Applied Artificial Intelligence
Visvesvaraya National Institute of Technology, Nagpur