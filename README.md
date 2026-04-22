# Credit Risk Analyzer 💳

A complete machine learning project for predicting credit risk using multiple supervised and unsupervised learning techniques, with FastAPI backend, Streamlit frontend, and Docker deployment.

## Project Overview

**Objective:** Predict whether a customer will default on their credit based on their financial profile.

**Dataset:** German Credit Dataset (1000 records, 20 features)

## Features & Models

### Machine Learning Models
- **Logistic Regression** - Baseline classification model
- **Random Forest** - Ensemble learning for better accuracy
- **K-Means Clustering** - Customer segmentation by risk profile
- **PyTorch Neural Network** - Deep learning alternative

### Key Features
- Customer age, income, credit amount, employment duration
- Credit history, purpose, savings status
- Employment status, property ownership, phone availability
- Previous credit performance, additional debtors status

## Tech Stack

```
Backend:      FastAPI + Uvicorn
Frontend:     Streamlit
ML:           Scikit-learn, PyTorch, Pandas, Numpy
Database:     Pickle/Joblib (model storage)
Deployment:   Docker, Docker Compose
VCS:          Git
CI/CD:        GitHub Actions
```

## Quick Start

### 1. Clone & Setup
```bash
git clone <repo-url>
cd CreditRiskAalyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Locally

**Start Backend:**
```bash
cd app
uvicorn main:app --reload
```

**Start Frontend (new terminal):**
```bash
streamlit run streamlit_app.py
```

### 3. Docker Deployment
```bash
docker-compose up
```

Access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Frontend: http://localhost:8501

## Project Structure

```
CreditRiskAalyzer/
├── notebooks/
│   └── basic.ipynb              # EDA & data exploration
├── src/
│   ├── preprocessing.py         # Data preprocessing
│   ├── feature_engineering.py   # Feature creation
│   ├── models.py                # ML models (LR, RF)
│   ├── clustering.py            # K-Means clustering
│   └── neural_network.py        # PyTorch network
├── app/
│   ├── main.py                  # FastAPI application
│   ├── schemas.py               # Pydantic models
│   ├── routes.py                # API endpoints
│   └── chatbot.py               # Rule-based chatbot
├── models/                      # Saved trained models
├── streamlit_app.py             # Streamlit frontend
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Multi-container setup
├── .github/workflows/           # CI/CD pipeline
├── README.md                    # This file
├── EXECUTION_PLAN.md            # Day-by-day plan
└── .gitignore

```

## Execution Plan

Follow the **25-day execution plan** in `EXECUTION_PLAN.md` for step-by-step guidance.

**Current Status:** ✅ Day 1 Complete (EDA & Data Understanding)

## Installation & Dependencies

See `requirements.txt`:
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=1.9.0
fastapi>=0.68.0
uvicorn>=0.15.0
streamlit>=1.0.0
pydantic>=1.8.0
joblib>=1.0.0
```

## API Endpoints (FastAPI)

### Predictions
- `POST /predict` - Single customer prediction
- `POST /predict_batch` - Batch predictions

### Model Info
- `GET /models/list` - Available models
- `GET /models/compare` - Compare models
- `POST /explain` - Feature importance

### Chat
- `POST /chat` - Chatbot queries

Full documentation: `http://localhost:8000/docs`

## Model Performance (Expected)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~72% | 0.70 | 0.68 | 0.69 |
| Random Forest | ~75% | 0.74 | 0.71 | 0.72 |
| Neural Network | ~73% | 0.72 | 0.70 | 0.71 |

## Clustering Results

K-Means identifies customer segments:
- **Segment 0:** Low risk, stable credit
- **Segment 1:** Medium risk, moderate credit
- **Segment 2:** High risk, problematic credit

## Learning Outcomes

By completing this project, you'll learn:
- ✅ Full ML pipeline from data to deployment
- ✅ Multiple ML algorithms (supervised & unsupervised)
- ✅ Deep learning with PyTorch
- ✅ Backend API development with FastAPI
- ✅ Frontend development with Streamlit
- ✅ Docker containerization
- ✅ CI/CD automation
- ✅ Git workflows

## Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [PyTorch Tutorial](https://pytorch.org/tutorials/)
- [Docker Basics](https://docs.docker.com/)

## Troubleshooting

**Port already in use?**
```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>
```

**Import errors?**
```bash
pip install --upgrade requirements.txt
python -m pip install --upgrade pip
```

## Contributing

This is a learning project. Feel free to improve any component!

## License

MIT License - Use freely for learning

---

**Author:** Your Name  
**Created:** April 2026  
**Status:** In Development (Follow EXECUTION_PLAN.md)

For questions or issues, refer to the EXECUTION_PLAN.md for detailed day-by-day guidance.
