---
title: RiskSight Pro
sdk: docker
pinned: false
---

# 🛡️ RiskSight Pro
### Banking & Insurance Risk Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-orange?logo=scikitlearn)
![Plotly](https://img.shields.io/badge/Plotly-5.22-purple?logo=plotly)
![Docker](https://img.shields.io/badge/Docker-Hugging%20Face%20Spaces-yellow?logo=docker)

> A professional risk analytics dashboard built with Flask, Scikit-learn, and Plotly — covering core risk domains in **banking** and **insurance**. Deployed on Hugging Face Spaces via Docker.

---

## 🚀 Live Demo

👉 [huggingface.co/spaces/yourname/risksight-pro](https://huggingface.co/spaces/yourname/risksight-pro)

---

## 📸 Features at a Glance

| Module | Risk Concept | ML/Stats Method |
|---|---|---|
| Credit Risk | PD · LGD · EAD · Expected Loss | Random Forest |
| Fraud Detection | Transaction anomaly scoring | Gradient Boosting |
| Market Risk | VaR · CVaR · Sharpe · Drawdown | Historical Simulation |
| Loan Portfolio | Concentration Risk · Credit Grade Mix | EL = PD × LGD × EAD |
| Claims Analytics | Claims frequency & severity | Descriptive statistics |
| Underwriting Risk | Risk loading · Premium pricing | Logistic Regression |
| Loss Ratio | Combined ratio · Profitability | Actuarial analysis |

---

## 🏗️ Project Structure

```
risksight-pro/
├── app.py              # Flask app — routes, models, synthetic data, charts
├── requirements.txt    # Python dependencies
├── Dockerfile          # HuggingFace Spaces Docker config
└── README.md
```

---

## 🧠 Risk Concepts Covered

### Banking
- **Probability of Default (PD)** — likelihood a borrower will fail to repay
- **Loss Given Default (LGD)** — estimated loss if a borrower defaults
- **Exposure at Default (EAD)** — total outstanding amount at time of default
- **Expected Loss (EL)** — `EL = PD × LGD × EAD` (Basel III Pillar 1)
- **Value at Risk (VaR)** — maximum daily loss at 95% and 99% confidence
- **Conditional VaR / Expected Shortfall (CVaR)** — average loss beyond VaR threshold
- **Sharpe Ratio** — risk-adjusted return metric
- **Concentration Risk** — over-exposure to a single sector or product

### Insurance
- **Loss Ratio** — `Claims Paid ÷ Premiums Earned` (key profitability metric)
- **Combined Ratio** — `Loss Ratio + Expense Ratio` (underwriting profitability)
- **Underwriting Risk** — identifying high-risk applicants before policy issuance
- **Risk Loading** — premium surcharge applied to high-risk policyholders
- **Claims Severity & Frequency** — size and rate of insurance claims

---

## ⚙️ Tech Stack

- **Backend** — Flask, Pandas, NumPy
- **Machine Learning** — Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- **Visualization** — Plotly (server-side JSON, rendered client-side)
- **Frontend** — Bootstrap 5, Font Awesome, vanilla JS
- **Data** — Fully synthetic, generated with NumPy/Pandas (no real customer data)
- **Deployment** — Docker on Hugging Face Spaces (port 7860)

---

## 🏃 Run Locally

```bash
# 1. Clone
git clone https://github.com/yourname/risksight-pro.git
cd risksight-pro

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py

# Visit → http://localhost:7860
```

### Or with Docker

```bash
docker build -t risksight-pro .
docker run -p 7860:7860 risksight-pro
```

---

## ☁️ Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Push your files:

```bash
git remote add hf https://huggingface.co/spaces/yourname/risksight-pro
git push hf main
```

The Space will build automatically and be live in ~2 minutes.

---

## 📌 Important Notes

- All data in this dashboard is **100% synthetic** — generated programmatically for demonstration purposes only. No real customer, financial, or personal data is used.
- This project is intended as a **portfolio demonstration** of data science and risk analytics skills.
- ML models are trained on synthetic data at startup and are **not production-grade**.

---

## 👤 Author

**Mohammad Noorchenarboo**
Data Scientist | Artificial Intelligence Researcher · Ontario, Canada

- 📧 [mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)
- 💼 [linkedin.com/in/mnoorchenar](https://www.linkedin.com/in/mnoorchenar)
- 🌐 [mnoorchenar.github.io](https://mnoorchenar.github.io/)
- 🤗 [huggingface.co/mnoorchenar](https://huggingface.co/mnoorchenar/spaces)
- 🎓 [Google Scholar](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.