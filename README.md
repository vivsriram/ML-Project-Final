# TSA Checkpoint Throughput Forecasting

Predicts hourly passenger volume at TSA security checkpoints across 20 major U.S. airports. A LightGBM model trained on TSA throughput data (2019–2024) and BTS flight schedule data achieves R² = 0.85 and MAE = 124 passengers per checkpoint-hour on the held-out test set — a 16% improvement over a naive historical-mean baseline.

The model is deployed as a Streamlit web app on AWS EC2, live at **http://3.85.169.163:8501**.

**Team:** Arend Colle, Vivek Sriram, Tiffany T. Nguyen

---

## Repo Structure

```
├── notebooks/
│   └── 03_modelTraining_final.ipynb   # Data cleaning, EDA, model training, evaluation
├── app/
│   └── streamlit_app.py               # Streamlit prediction interface (deployed on EC2)
├── scripts/
│   ├── download_bts_data.py           # Downloads BTS on-time performance data (2022–2025)
│   ├── generate_aws_diagram.py        # Generates AWS architecture diagram (PNG)
│   └── train_throughput_model.py      # Standalone training script (legacy)
├── models/                            # Trained model artifacts — download from S3 (see below)
├── visuals/                           # Generated charts and diagrams
├── docs/                              # Project memos and proposal
├── data/
│   ├── throughput/                    # TSA checkpoint throughput CSVs — committed (one per airport, ~3 MB each)
│   ├── bts/                           # BTS schedule/load factor CSVs — committed (one per year, ~22 MB each)
│   ├── ontime/                        # BTS on-time performance CSVs — gitignored (600 MB/year; see Data section)
│   └── cached/                        # Preprocessed feature matrix — gitignored (243 MB; download from S3)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.10+
- AWS CLI configured with credentials that have S3 read access

### Install dependencies
```bash
pip install -r requirements.txt
```

### Environment variables
```bash
cp .env.example .env
# Edit .env — set AWS_DEFAULT_REGION and S3_BUCKET if different from defaults
```

---

## Data

Raw data files are not committed to this repo (too large). To reproduce the pipeline from scratch:

### TSA Throughput Data
Download from [TSA's public data portal](https://www.tsa.gov/travel/passenger-volumes) — one CSV per airport, place in `data/throughput/`.

### BTS On-Time Performance Data
Run the download script (scrapes BTS, ~2 hours for 2022–2025):
```bash
python scripts/download_bts_data.py
```
Output CSVs land in `data/ontime/`.

### Preprocessed feature cache
The notebook produces `data/cached/modelDf.csv` during the feature engineering step. It is also available in S3:
```bash
aws s3 cp s3://tsa-throughput-model/modelDf.csv data/cached/modelDf.csv
```

---

## Running the Pipeline

Run steps in order from the **repo root**.

### 1. Train the model
Open and run `notebooks/03_modelTraining_final.ipynb` top-to-bottom. The notebook:
- Loads and merges TSA throughput + BTS on-time data
- Engineers features (hour, day-of-week, month, holiday proximity, flight schedule)
- Runs a grid search over LightGBM hyperparameters
- Trains the final model and evaluates on the held-out test set
- Saves `models/lgbmModel.pkl`

> **Note:** Run the notebook from the repo root (`jupyter notebook` from `/ml-project`), not from inside `notebooks/`, so relative data paths resolve correctly.

### 2. Upload artifacts to S3
```bash
aws s3 cp models/lgbmModel.pkl s3://tsa-throughput-model/lgbmModel.pkl
aws s3 cp data/cached/modelDf.csv s3://tsa-throughput-model/modelDf.csv
```

### 3. Run the Streamlit app locally
```bash
# Download model from S3 first (or use locally trained model)
aws s3 cp s3://tsa-throughput-model/lgbmModel.pkl models/lgbmModel.pkl

streamlit run app/streamlit_app.py
```
Opens at http://localhost:8501.

---

## AWS Deployment

The app runs on an EC2 t3.medium instance (Amazon Linux 2023) behind a public IP. S3 stores the model artifact and data cache; EC2 loads them at startup and serves live predictions.

### One-time EC2 setup
```bash
# SSH into instance
ssh -i your-key.pem ec2-user@<EC2_PUBLIC_IP>

# Install dependencies
sudo dnf install -y python3-pip git
pip3 install -r requirements.txt

# Clone repo and configure environment
git clone https://github.com/vivsriram/ml-project.git
cd ml-project
cp .env.example .env   # edit with your S3 bucket name

# Launch app (persists after SSH disconnect)
nohup streamlit run app/streamlit_app.py --server.port 8501 &
```

### Restarting after AWS Academy session timeout
AWS Academy Learner Lab sessions time out periodically, which stops the EC2 instance.

1. Log into AWS Academy → EC2 → select instance → **Start**
2. Wait ~30 seconds, then SSH in
3. Re-run: `cd ml-project && nohup streamlit run app/streamlit_app.py --server.port 8501 &`
4. Verify: http://\<EC2_PUBLIC_IP\>:8501

### Credentials
The EC2 instance uses an IAM role with S3 read access — no credentials are stored on the instance or in the repo. If running locally, configure AWS credentials via `aws configure` or environment variables.

---

## Model Performance

| Metric | Naive Baseline | LightGBM |
|---|---|---|
| R² | 0.7968 | 0.8509 |
| MAE (passengers) | 147.07 | 123.85 |
| RMSE | 213.53 | 182.93 |
| Daily Pearson r | — | 0.9398 |

Test set: final 60 days of data, 159,021 records. Naive baseline = historical mean per (airport, checkpoint, hour, day-of-week, month).

---

## References

Monmousseau, P., Marzuoli, A., Feron, E., & Delahaye, D. (2020). Predicting and controlling airport passenger flow. *Transportation Research Part C: Emerging Technologies, 120*, 102796.
