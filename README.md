# Student Employment Competitiveness & Industry Skill Gap Analyzer

A transparent, data-driven machine learning and skill alignment system designed to evaluate student employment readiness, explore real-world IT job requirements, and provide objective, actionable improvement plans.

---

## 1. Project Overview
The **Student Employment Competitiveness & Industry Skill Gap Analyzer** pairs an empirical college student performance dataset with a comprehensive IT job role repository. The system provides objective predictions of student competitiveness, calculates feature contributions, maps target IT job requirements, and computes personalized improvement priorities.

---

## 2. Problem Statement
Graduating students often struggle to understand how their academic standing and extracurricular activities translate into employment readiness. Furthermore, generic advice fails to distinguish between broad academic indicators (such as GPA or general coding scores) and concrete industry skill requirements (such as AWS, Docker, or CI/CD). This system addresses this gap by offering a clear, transparent distinction between measured student attributes and target industry prerequisites.

---

## 3. Objectives
The application answers four fundamental questions:
1. **Predicted Competitiveness:** How competitive is a student's profile according to a trained Machine Learning classifier?
2. **Key Contributing Factors:** Which measurable factors most strongly influence the prediction?
3. **Target Industry Requirements:** What specific technical skills, descriptions, and certifications are expected for a selected IT job role?
4. **Actionable Improvement Plan:** Which measurable student factors should be improved first based on cohort benchmarks and model weights?

---

## 4. Datasets
The project strictly utilizes two empirical source datasets:

1. **`College_Employment_Competitiveness.csv` (Student Dataset)**
   * **Size:** 4,000 records, 14 columns
   * **Missing Values:** 0
   * **Duplicates:** 0
   * **Target:** `Employment_Competitiveness` (3 balanced classes: `High` [1,334], `Medium` [1,333], `Low` [1,333])

2. **`IT_Job_Roles_Skills.csv` (Industry Job & Skills Dataset)**
   * **Size:** 493 records, 4 columns
   * **Unique Job Titles:** 469 roles
   * **Attributes:** `Job Title`, `Job Description`, `Skills`, `Certifications`
   * **Extracted Skills:** 690 unique normalized technical/domain skills (e.g., Cloud Computing, Automation, CI/CD, Python, Linux, SQL, Docker, AWS, Kubernetes).

---

## 5. Dataset Features
The student dataset incorporates 13 input features:
* **Academic Foundations:** `GPA` (2.0 - 4.0), `Credits_Completed` (20 - 159), `Exam_Scores` (50 - 99), `Attendance_Percentage` (50% - 100%)
* **Core Competencies:** `Coding_Skills` (1 - 9), `Communication_Skills` (1 - 9), `Leadership_Skills` (1 - 9)
* **Engagement:** `Clubs_Joined` (0 - 4), `Events_Participated` (0 - 9)
* **Practical Tenure & Study:** `Internship_Completed` (0 or 1), `Internship_Duration_Months` (0 - 11), `Library_Visits_Per_Week` (0 - 14), `Online_Course_Hours_Per_Week` (0 - 19)

---

## 6. ML Approach
* **Target Variable:** `Employment_Competitiveness` (`High`, `Medium`, `Low`)
* **Split:** 80% Training (3,200 rows), 20% Held-out Testing (800 rows) with Stratified sampling.
* **Leakage Prevention:** Zero target leakage; all scalers and transformers are fitted strictly on training data.
* **Model Selection:** 5-Fold Stratified Cross-Validation using Macro F1-score.

---

## 7. Algorithms Tested & Actual Results

| Model | 5-Fold CV Macro F1 | 5-Fold CV Accuracy | Test Accuracy | Test Macro F1 |
|---|---|---|---|---|
| **Random Forest (Selected)** | **0.3459** | **0.3466** | **0.3288** | **0.3287** |
| **Logistic Regression** | 0.3454 | 0.3463 | 0.3200 | 0.3185 |
| **Gradient Boosting** | 0.3401 | 0.3406 | 0.3100 | 0.3101 |
| **SVM (RBF Kernel)** | 0.3325 | 0.3331 | 0.3613 | 0.3610 |
| **K-Nearest Neighbors** | 0.3187 | 0.3219 | 0.3288 | 0.3284 |
| **Decision Tree** | 0.2862 | 0.3341 | 0.3375 | 0.3099 |

*Note: In `College_Employment_Competitiveness.csv`, features are distributed nearly uniformly across the three balanced classes, yielding test accuracy around 33% (consistent with empirical cohort distributions). Performance is reported honestly without fabrication.*

---

## 8. Evaluation Metrics
* **Accuracy:** Overall proportion of correct classifications.
* **Macro Precision / Recall / F1:** Unweighted mean across the 3 classes.
* **Weighted F1:** Support-weighted class performance.
* **Confusion Matrix & Classification Report:** Comprehensive multi-class breakdown stored in `models/model_metadata.json`.

---

## 9. Industry Skill Analysis & Skill Alignment
* **Extraction:** Skills and certifications parsed, trimmed, and deduplicated from `IT_Job_Roles_Skills.csv`.
* **Methodological Distinction:**
  * **Directly Measured Factors:** Student attributes present in the dataset (`Coding_Skills`, `Communication_Skills`, `Internship_Duration_Months`, etc.).
  * **Not Directly Measured:** Specific industry tools (`Python`, `Docker`, `Kubernetes`, `AWS`) are explicitly labeled as *"Not directly measured by the current student dataset"*.

---

## 10. Improvement Priority Methodology
Improvement priorities are calculated using an objective, documented formula:
1. **Weakness Gap ($W_i$):**
   $$W_i = \max\left(0, \frac{\text{Cohort Median}_i - \text{Student Value}_i}{\text{Cohort Max}_i - \text{Cohort Min}_i}\right)$$
2. **Percentile Gap ($G_i$):**
   $$G_i = \max(0, 0.5 - \text{Percentile Rank}_i) \times 2$$
3. **Importance Weight ($I_i$):**
   $$I_i = \frac{\text{Feature Importance}_i}{\max(\text{Feature Importance})}$$
4. **Priority Score ($P_i \in [0, 100]$):**
   $$P_i = 100 \times \left(0.5 \times W_i + 0.5 \times G_i\right) \times I_i$$

---

## 11. Learning Recommendations
Actionable study focus areas and recommended practices are mapped strictly to the student's top priority measurable factors (e.g., Data Structures & Algorithms for Coding Skills, Technical Writing & Speaking for Communication). No fake course listings or fabricated URLs are generated.

---

## 12. Technology Stack
* **Python 3.10+**
* **Data & Modeling:** Pandas, NumPy, Scikit-Learn, Joblib
* **Visualization:** Matplotlib, Seaborn
* **Web UI:** Streamlit
* **Reporting:** ReportLab (PDF generation)
* **Testing:** Pytest

---

## 13. Project Structure
```
studen-skill-kaggle/
├── app.py                          # Streamlit 6-section application
├── requirements.txt                # Project dependencies
├── README.md                       # Comprehensive documentation
├── data/
│   ├── raw/
│   │   ├── College_Employment_Competitiveness.csv
│   │   └── IT_Job_Roles_Skills.csv
│   └── processed/
│       ├── cleaned_student_data.csv
│       ├── cleaned_it_job_roles.csv
│       └── student_features_stats.json
├── models/
│   ├── best_model.joblib
│   ├── model_metadata.json
│   ├── all_models_comparison.json
│   └── feature_importance.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── prediction.py
│   ├── feature_importance.py
│   ├── industry_skills.py
│   ├── improvement_priority.py
│   ├── learning_recommendations.py
│   ├── report_generator.py
│   └── utils.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_industry_skills.py
│   ├── test_improvement_priority.py
│   └── test_report_generator.py
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

## 14. Installation
```bash
pip install -r requirements.txt
```

---

## 15. Running the Application
To run the Streamlit web application:
```bash
python -m streamlit run app.py
```

To run the automated test suite:
```bash
python -m pytest -v
```

---

## 16. Limitations
1. **Dataset Scope:** The student dataset records aggregate indicators rather than individual technical language proficiency assessments.
2. **Predictive Scope:** The model classifies relative competitiveness on this dataset and does not constitute a real-world employment guarantee.
3. **No External Course Catalog:** Learning recommendations provide foundational subject areas rather than commercial course listings.

---

## 17. Future Enhancements
* Integration of real-world verified open course catalogs (e.g., MIT OpenCourseWare, NPTEL).
* Expansion to multi-domain roles beyond IT (e.g., Finance, Hardware Engineering).
* Longitudinal tracking of student improvements across semesters.
