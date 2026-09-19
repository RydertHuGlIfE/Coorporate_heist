# The Corporate Heist - Solution Documentation
Team name: Kestrel Alpha
Unstop team ID: CH-2026-919
Members: Pradeepto & Team

---

## 1. Summary
Our solution identifies the top 500 candidates most likely to be Nightingale's top 5% performers by combining a high-capacity machine learning ranking engine with a strict rules-based insider heuristics filter. We train an ensemble of LightGBM Regressors (predicting raw 1-year post-hire performance) and LightGBM Classifiers (targeting the top 5% threshold directly) on 20,000 historical hires. Crucially, we adjust for the current hiring cycle by incorporating public code contributions, rewarding non-traditional institutes when paired with top technical assessment scores and role fit, and shrinking residuals to honor the "old-boys' network" colleges. Finally, we execute rigorous data integrity pipelines: removing fabricated profiles (timeline contradictions and accelerated title claims), excluding candidates with >60 days notice, and deduplicating multiple recruiter submissions using Union-Find to ensure each individual appears at most once on the shortlist.

---

## 2. Data Cleaning and Parsing
Recruiter-entered fields contained massive formatting inconsistencies and noisy inputs:
- **Total Experience (`total_experience`)**: Extracted numeric years across diverse formats: `"17.6 years"`, `"8+ yrs"`, `"Apprentice [23 mo]"`, `"<1 year"`. Normalized months to fractional years (`/12`).
- **Assessment & Aptitude Scores (`technical_assessment`, `aptitude_score`)**: Normalized variable representations including percentages (`"75%"` -> 75), explicit denominators (`"71/100"`, `"5.7/10"` -> 57), and handled missing codes (`"not taken"`, `"absent"`, `"na"` -> NaN).
- **Ratings & KPIs (`last_rating`, `kpi_met`)**: Mapped rating strings (`"outstanding"`, `"exceeds expectations"`, `"5/5"`) to a continuous 1–5 scale, and KPI flags (`"Y"`, `"yes"`, `"1"`) to binary values.
- **CTC Normalization (`current_ctc`, `expected_ctc`)**: Standardized disparate recruiter currency inputs into uniform Lakhs INR per annum (e.g. converting USD at 83/1e5, `"7.53 Cr"` -> 753 Lakhs, `"₹30,46,000"` -> 30.46 Lakhs). Computed the expected CTC hike ratio.
- **Notice Period (`notice_period`)**: Parsed strings (`"15 days"`, `"1 month"`, `"Immediate joiner"`, `"Serving notice - 30 days"`) into numeric calendar days.
- **Career Path (`career_path`)**: Deconstructed trajectory steps (`"Title [duration]"`) using regular expressions to extract total years, current seniority level (mapped via seniority keyword hierarchy from Intern:0 to CTO:9), time in last role, and total career promotions.
- **Institute Canonicalization (`institute`)**: Normalized case, removed punctuation, consolidated regional abbreviations (e.g. `"iitb"` -> `"iit bombay"`, `"iitm"` -> `"iit madras"`, `"trichy"` -> `"tiruchirappalli"`, unified BITS, DTU, NSUT, and IISc).

---

## 3. What Drives a Great Hire — Your Findings
Analysis of the 20,000 historical records (`train.csv`) and verification on the Ledger (`dev.csv`) revealed key patterns:
1. **Technical Assessment & Role Fit Interaction**: Technical assessment is the strongest single predictor, but its predictive power is amplified when combined with role-specific skill alignment. Candidates whose skills directly match their applied role's skill distribution score significantly higher.
2. **Career Stability & Promotion Velocity**: Steady promotions without erratic gaps correlate strongly with high post-hire scores. Candidates with consistent tenure per employer outperform frequent job-hoppers.
3. **Recruiter Note Nuances**: Text extraction revealed distinct behavioral signals. Candidates associated with on-call ownership, cross-team initiatives, and system design commendations had a ~7.0% top-performer rate (versus the 5.0% base rate), whereas notes citing debugging struggles or missed sprint commitments dropped below 3.8%.
4. **Historical College Residuals**: Shrunk empirical Bayes residuals showed a small persistent advantage for a core group of premier institutes (IISc, IIT Kharagpur, BITS Pilani, IIT Kanpur, IIT Bombay, NSUT), corroborating the "old-boys' network" insider clue.

---

## 4. The Current Cycle
The current cycle (the Vault, `test.csv`) presents a distribution shift driven by updated leadership hiring mandates:
1. **Public Code Contributions (`public_code_contributions`)**: This feature was not present in historical data. In the test set, ~5,552 candidates have tracked contributions (median 6, top 5% >= 42). Candidates with dozens of merged contributions (>= 24) receive a fast-track bonus, rewarding sustained open-source output.
2. **De-emphasis of General College Prestige**: While general tier-1 status is no longer universally rewarded, candidates from institutes *never previously hired from* who achieved top-decile technical assessment scores and high role fit receive an explicit boost.
3. **Preservation of the Old-Boys' Network**: Shrunk historical residuals for Nightingale leadership's core colleges are retained as a calibrated additive signal.

---

## 5. Profiles You Excluded and Why
We enforce strict elimination rules prior to final ranking:
1. **Notice Period > 60 Days (Clue 3)**: Anyone unable to join within 2 months is excluded (`notice_period > 60 days`). This disqualified 92 candidates in the test set.
2. **Fabricated Career Progression (Clue 2)**: Candidates whose current title level (e.g. Lead, Architect, Director, VP) was attained faster than the 1st percentile of minimum historical experience for that title tier. This eliminated 137 implausibly inflated profiles.
3. **Timeline Contradictions (Clue 6)**: Profiles with impossible timelines (e.g. claimed total experience exceeding years since graduation by >3.5 years, or experience accumulated before age 16) were purged as fabricated resumes (98 instances).
4. **Deduplication of Candidate Profiles (Clue 6)**: The same applicant frequently appeared under multiple candidate IDs entered by different recruiters. We constructed a Union-Find disjoint-set graph over exact 10-digit phone numbers, normalized emails, and name + graduation year keys. Out of 246 duplicate groups in `test.csv`, only the single highest-scoring profile was retained.

---

## 6. Model and Selection Procedure
- **Feature Set**: 135 features spanning clean numerical attributes, composite assessment scores, seniority metrics, career progression velocity, CTC hike dynamics, role-fit affinities, skill presence flags, and recruiter note sentiment n-grams.
- **Model Architecture**:
  - **Ensemble Regressor**: 4 LightGBM Regressors trained across different seeds predicting continuous `post_hire_score`.
  - **Ensemble Classifier**: 4 LightGBM Classifiers trained across different seeds with positive class weighting (`scale_pos_weight=2.2`) predicting the binary top 5% outcome.
- **Score Formulation**:
  $$\text{Base Score} = 0.55 \times \text{Rank}(\hat{y}_{\text{reg}}) + 0.45 \times \text{Rank}(\hat{p}_{\text{clf}})$$
  Standardized $\text{Score} = Z(\text{Base Score}) + \text{Bonus}_{\text{contrib}} + \text{Bonus}_{\text{new\_college\_strong}} + \text{Bonus}_{\text{old\_boys}}$.
- **Shortlist Selection**: All candidates with exclusion flags receive $-\infty$. The remaining candidates are deduplicated greedily by descending score, selecting exactly the top 500 unique individuals.

---

## 7. What Did Not Work
- **Naive Full-Feature Ensembles with Metro/Prestige Proxies**: Training blindly on candidate city, degree prestige, and recruiter referral channel learned historical selection bias that degraded performance on newer cycles where those rules were discarded.
- **Pure Regression Ranking**: Using only regression on `post_hire_score` yielded lower top-5% precision (overlap of 30.8%) compared to our dual regression + classification blend (overlap of 35.5%, dev precision 52.7%).
- **Over-weighting Code Contributions**: Linearly scaling code contributions caused excessive displacement of top-tier technical performers by low-scoring applicants with high commits. Calibrating the bonus as an additive boost only above sustained thresholds (>= 24) preserved assessment quality.

---

## 8. External Resources and AI Tools
- **Libraries**: `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `scipy` (all compliant with competition Appendix A).
- **AI Tools**: Claude / Antigravity pair programming assistant used for exploratory data analysis, rapid prototyping of feature extractors, and formatting verification.

---

## 9. How to Run
```bash
# Environment: Python 3.11 / Linux (CPU only, ~60-90s execution time)
python3 main.py

# Verify format
python3 check_format.py submission.csv test.csv
```
- **Random Seed**: Fixed at `20260919`.
- **Outputs**: Generates compliant `submission.csv` with exactly 500 uniquely ranked candidates.
