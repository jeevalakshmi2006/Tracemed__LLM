# 🫀 TraceMed AI
### Explainable Medical Risk Prediction & Clinical Decision Support System


## 🧠 What is TraceMed AI?

Most AI healthcare systems predict disease risk but act as a **"black box"** — doctors see a number but have no idea why.

**TraceMed AI solves this.**

It predicts heart disease risk AND explains exactly which patient factors caused the prediction, in language both doctors and patients can understand.

```
Normal AI Model Output:       TraceMed AI Output:
                              
Heart Disease Risk = 91%      Heart Disease Risk = 91%
                              
That's all.                   Top Contributing Factors:
                              1. Major Vessels       +28.9%
                              2. Thal                +20.0%
                              3. Max Heart Rate      +11.1%
                              
                              AI Clinical Explanation:
                              "This patient presents a high
                              cardiovascular risk profile..."
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🌲 **Random Forest** | Built completely from scratch — no scikit-learn |
| 🔍 **Explainability Engine** | Permutation importance built from scratch |
| 🤖 **LLM Integration** | Llama 3 generates natural language clinical explanations |
| 📊 **Interactive Dashboard** | Risk gauge, contribution charts, real-time predictions |
| 🔬 **What-If Simulator** | Change patient values and see risk change instantly |
| 👨‍⚕️ **Dual View** | Separate doctor and patient explanation modes |

---

## 🏗️ System Architecture

```
Patient Data Input
       ↓
Data Preprocessing (normalization, train/test split)
       ↓
Random Forest Model (50 trees, built from scratch)
       ↓
Risk Prediction Score (0–100%)
       ↓
Permutation Importance Engine (built from scratch)
       ↓
Top Contributing Features
       ↓
LLM Explanation Layer (Llama 3 via Groq API)
       ↓
Streamlit Doctor Dashboard
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Machine Learning** | Random Forest — built from scratch |
| **Explainability** | Permutation Importance — built from scratch |
| **LLM** | Llama 3 (8B) via Groq API (free) |
| **Dashboard** | Streamlit |
| **Charts** | Plotly |
| **Dataset** | UCI Cleveland Heart Disease (207 patients) |

> ⚠️ No ML libraries used (no scikit-learn, no SHAP). Everything built from scratch in pure Python.

---

## 📁 Project Structure

```
tracemed_ai/
├── data/
│   └── heart.csv                  # UCI Cleveland Heart Disease dataset
├── utils/
│   ├── preprocessing.py           # Data cleaning, normalization, train/test split
│   ├── decision_tree.py           # Decision Tree built from scratch
│   ├── random_forest.py           # Random Forest built from scratch
│   ├── feature_importance.py      # Permutation importance from scratch
│   └── llm_explanation.py         # Groq API + fallback explanation
├── ui/
│   └── dashboard.py               # Full Streamlit dashboard
├── main.py                        # Backend test runner
├── .env                           # API keys (not committed)
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/jeevalakshmi2006/Tracemed__LLM.git
cd Tracemed__LLM
```

### 2. Install dependencies
```bash
pip install streamlit plotly pandas numpy requests fpdf2 python-dotenv
```

### 3. Set up API key (optional)
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: https://console.groq.com

> If no API key is provided, the system uses a built-in fallback explanation engine.

### 4. Run the dashboard
```bash
streamlit run ui/dashboard.py
```

### 5. Test the backend only
```bash
python main.py
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | 80.95% |
| **Precision** | 91.30% |
| **Recall** | 77.78% |
| **F1 Score** | 84.00% |

Trained on 165 samples, tested on 42 samples from the UCI Cleveland Heart Disease dataset.

---

## 🖥️ Dashboard Preview

### Tab 1 — Risk Analysis
- Risk score gauge (0–100%)
- Feature contribution bar chart (red = increases risk, green = decreases risk)
- AI-generated doctor explanation
- AI-generated patient-friendly explanation

### Tab 2 — What-If Simulator
- Adjust blood pressure, cholesterol, heart rate and more
- See risk change in real time with before/after gauges
- Understand how lifestyle changes affect heart disease risk

### Tab 3 — Model Insights
- Global feature importance across all patients
- Full model performance metrics
- Technology overview

---

## 🧩 Core Modules Explained

### `utils/decision_tree.py`
Implements a full Decision Tree from scratch using:
- Gini impurity for split quality
- Recursive tree building with max depth and min samples stopping criteria
- Probability prediction at leaf nodes

### `utils/random_forest.py`
Implements Random Forest from scratch using:
- Bootstrap sampling (random rows with replacement)
- Random feature selection per split (√n features)
- Majority voting across 50 trees for final prediction

### `utils/feature_importance.py`
Implements Permutation Importance from scratch:
- Shuffles one feature at a time
- Measures accuracy drop — bigger drop = more important feature
- Also computes patient-level contributions vs training mean

### `utils/llm_explanation.py`
- Builds structured prompts from feature contributions
- Calls Llama 3 via Groq API for clinical explanation
- Falls back to rule-based explanation if API unavailable

---

## 📋 Dataset

**UCI Cleveland Heart Disease Dataset**
- 207 patients (after cleaning)
- 13 features: Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, ST Depression, Slope, Major Vessels, Thal
- Binary target: 0 = No Heart Disease, 1 = Heart Disease

---
