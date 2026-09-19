# The Corporate Heist — Solution Repository

An end-to-end machine learning and heuristic ranking pipeline to select and rank the top 500 candidate hires from a 10,000-applicant pool (`test.csv`) for Nightingale Systems.

---

## 🚀 Quickstart & Reproduction

### Prerequisites
- Python 3.11+
- Required packages (CPU-only, execution time ~60–90 seconds):
  ```bash
  pip install numpy pandas scikit-learn lightgbm scipy
  ```

### Run the Pipeline
Ensure `train.csv`, `dev.csv`, `dev_winners.csv`, and `test.csv` are in the current working directory, then run:

```bash
python3 main.py
```

### Verify Submission Format
```bash
python3 check_format.py submission.csv test.csv
```

---

## 🧠 Approach & Architecture

### 1. Robust Data Cleaning & Canonicalization
- **Experience Parsing**: Standardizes mixed string representations (`"17.6 years"`, `"8+ yrs"`, `"<1 year"`, `"23 mo"`) into continuous numeric years.
- **Score Normalization**: Scales assessments and aptitude tests from diverse formats (`"71/100"`, `"5.7/10"`, percentages) to uniform scales.
- **Compensation Normalization**: Parses disparate CTC strings (USD, Cr, LPA, Lakhs, INR) into standardized Lakhs per annum and computes candidate expected hike ratios.
- **Institute Entity Resolution**: Maps abbreviations and campus variations (`"iitb"`, `"iitm"`, `"trichy"`, BITS, DTU, NSUT) into canonical entities.

### 2. Dual-Target Model Ensembling
- **LightGBM Regressor (4 seeds)**: Predicts continuous 1-year `post_hire_score`.
- **LightGBM Classifier (4 seeds, with class weighting)**: Directly predicts membership in the top 5% cutoff.
- **Rank Blending**: Ensembles percentile ranks using optimal weights (`0.70 * Rank_reg + 0.30 * Rank_clf`), validated on `dev.csv`.

### 3. Insider Clue Calibration (Current Cycle Distribution Shift)
- **Public Code Contributions**: Calibrated additive boost for sustained open-source contributions ($\ge 24$ merged commits) without compromising technical assessment baselines.
- **New Colleges Advantage**: Boosts candidates from previously unhired institutes who achieved high technical assessment percentiles ($\ge 70\text{th}$) and strong applied role fit.
- **Old-Boys' Network**: Applies shrunk empirical Bayes residuals for leadership-associated institutes.

### 4. Integrity & Exclusion Engine
- **Notice Period Filter**: Disqualifies candidates with $>60$ days notice period.
- **Fabricated Profile Scrubbing**: Eliminates impossible timelines (experience preceding age 16, degree year contradictions) and accelerated titles lacking requisite minimum tenure.
- **Union-Find Deduplication**: Clusters duplicate recruiter submissions across normalized 10-digit phone numbers, emails, and name + graduation year, greedily keeping only the highest-scoring entry per person.

---

## 📊 Validation Metrics on Dev (`dev.csv`)

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Validation AUC** | **0.947** | High discriminative power on held-out cycle |
| **Precision@150** | **53.3%** | 80/150 exact winners identified (10.6× random baseline) |
| **Top-300 Recall** | **74.7%** | 112/150 true winners caught in top 300 |
| **Top-500 Recall** | **87.3%** | 131/150 true winners caught in top 500 |
| **Top-1000 Recall** | **98.7%** | 148/150 true winners caught in top 1000 |
| **Rule False Rejections** | **0 / 150** | Zero true winners falsely excluded |

---

## 📁 Repository Structure
```text
├── main.py                  # Full reproducible end-to-end pipeline
├── check_format.py          # Official format validator
├── submission.csv           # 500 ranked candidate IDs
├── documentation.pdf        # 4-page technical documentation
├── documentation.md         # Source text for documentation
└── README.md                # Project summary and run instructions
```