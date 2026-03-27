# 🚗 Auto Insurance Accident Risk Prediction — State Farm
**Course:** MISY 641 – Data Mining | University of Delaware  
**Team:** Samuel Bartuska, Rahul Chauhan, Sumit Sachdeva  
**Tools:** Python, Decision Tree, Naive Bayes Classification

---

## 📌 Project Overview
Built a predictive classification model to estimate the probability of a driver being involved in an at-fault accident in the next policy term. The goal is to help State Farm — the largest U.S. auto insurer (~18–19% market share) — price premiums more fairly and competitively.

---

## ❓ Business Problem
Auto insurers must forecast future claims to set accurate premiums:
- Underestimating accident risk → unprofitable pricing
- Overestimating accident risk → losing customers to competitors

This project builds a model predicting `Accident_next_term` (1 = accident, 0 = no accident) from historical driver and policy data.

---

## 📊 Dataset & Features
- **Unit of analysis:** One driver–policy term
- **Target variable:** `Accident_next_term` (binary: 1 or 0)
- **Key features used:**
  - Driver history: age band, years licensed, prior at-fault accidents, violations
  - Vehicle: make/model, model year, body type, safety features
  - Policy: coverage limits, tenure, multi-policy flag
  - Usage: annual miles, garaging region (urban/suburban/rural)
  - Telematics (if available): harsh braking, late-night driving share

---

## 🧠 Models Built
| Model | Description |
|---|---|
| **Decision Tree** | Splits data using information gain (entropy). Tuned with node limits (60–110 nodes) to prevent overfitting. |
| **Naive Bayes** | Calculates class probability using Bayes' rule with independence assumption. Uses Laplace smoothing. |

Both models evaluated using **K-Fold Cross Validation (K=5)**.

---

## 📈 Evaluation Metrics & Targets
| Metric | Target |
|---|---|
| Accuracy | ≥ 75% |
| False Negatives | < 15% |
| False Positives | < 10% |

Metrics include: Accuracy, Precision, Recall, Confusion Matrix.  
Naive Bayes probability cutoff tuned below 0.5 (conservative bias toward predicting accidents).

---

## 🔑 Key Findings
- Decision Tree and Naive Bayes models compared side by side
- Class imbalance addressed via probability cutoff tuning and class-weighting
- Data leakage carefully prevented (no post-incident fields used)
- Fairness guardrails: no protected attributes (race, ethnicity) used

---

## 📁 Repository Structure
```
├── data/               # Sample/synthetic data (actual State Farm data is proprietary)
├── notebooks/          # Jupyter notebooks with model building and evaluation
├── report/             # Final written report (PDF)
└── README.md
```

---

## 🛠️ How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/insurance_risk_model.ipynb
```

---

## 📚 References
- Tan, Steinbach & Kumar – *Introduction to Data Mining (2nd ed.)*
- Obasi IC, Benson C. – *Evaluating ML Techniques for Traffic Accident Severity (Heliyon, 2023)*
- State Farm Market Share Data: ValuePenguin/NAIC (2025)
