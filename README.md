 Student Performance Prediction  
**EONVERSE AI Intern Screening Challenge**

## 🚀 Overview
This project focuses on predicting student academic success (pass/fail and final grades) using the [UCI Student Performance dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance). The goal is to build an end-to-end machine learning pipeline that:
- Cleans, explores, and preprocesses real student data from Portuguese high schools (over 30 attributes: demographics, grades, family, habits, etc.)
- Trains, tunes, and compares classification and regression models to predict student outcomes
- Provides actionable insights for teachers, counselors, or administrators to support at-risk students

**This repository contains:**  
- A complete Google Colab notebook showing the full ML workflow  
- A detailed PDF report with analysis, model results, and visuals  
- A demonstration video (link placeholder: _update before public release_)

---

## 🗃️ Dataset Description

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance)  
- **Scope:** Records of secondary school students with features including:
  - **Demographics:** Age, gender, address, parents' education and jobs, family size and status, etc.
  - **Academics:** Previous grades (G1, G2), final grade (G3), school and educational support, study time.
  - **Lifestyle & Social:** Free time, going out, alcohol use, absences, extracurricular activities, internet access, and more.
- **Prediction Targets:**
  - **Classification:** Pass/Fail (derived from G3)
  - **Regression:** Final grade (G3, 0–20 scale)

---

## ⚙️ ML Pipeline & Methods

1. **Data Preprocessing**
   - Handle missing values (minimal in dataset)
   - Encode categorical variables (one-hot or label encoding)
   - Scale numeric features (standardization/normalization)
   - Split train/test sets (e.g. 80/20 stratification)

2. **Exploratory Data Analysis**
   - Visualize distributions, correlations, and key relationships
   - Assess predictors’ importance with statistical metrics and tree-based feature importances

3. **Feature Selection**
   - Use domain knowledge, correlation analysis, and model-based importances
   - Focus on top features (previous grades, study time, family support, etc.)

4. **Modeling**
   - **Classification:** Evaluate Logistic Regression, Decision Trees, Random Forest, SVM, etc.
   - **Regression:** Linear Regression, Random Forest Regressor, and others for G3.
   - **Tuning:** Hyperparameter optimization (GridSearchCV, cross-validation)

5. **Evaluation**
   - **Classification:** Accuracy, Precision, Recall, F1-score, ROC-AUC for a balanced assessment
   - **Regression:** R², MAE, RMSE, and residual visualization

6. **Visualization**
   - Key plots: feature importances, confusion matrix, ROC curves, regression fits

---

## 📂 Repository Structure

```
.
├── Student Performance Regressor.ipynb   # Main ML pipeline notebook
├── report.pdf                           # Detailed project & results summary
├── demo_video.mp4 (or link)             # Project demonstration video
├── requirements.txt                     # Python dependencies (pandas, scikit-learn, etc.)
└── README.md                            # This file
```

---

## 🖥️ Quickstart

### 1️⃣ Requirements
- Python 3.8 or higher
- See `requirements.txt` (or: pandas, numpy, scikit-learn, matplotlib, seaborn)

### 2️⃣ Clone the repo
```sh
git clone https://github.com/Prakash-codeMaker/Eonverse-Screening-Challenge.git
cd Eonverse-Screening-Challenge
```

### 3️⃣ Environment Setup *(optional but recommended)*
```sh
python -m venv venv
source venv/bin/activate          # On Unix/macOS
# On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
*(Or manually: `pip install numpy pandas scikit-learn matplotlib seaborn`)*

### 4️⃣ Run the Notebook
- Open `Student Performance Regressor.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/)
- All steps from data load to prediction are included

### 5️⃣ (Optional) View the PDF Report and Demo Video for project summary and workflow demonstration.

---

## 🔍 Insights & Motivation

- **Why predict student performance?**  
  Early detection of at-risk students allows for personalized interventions (tutoring, counseling), improving educational outcomes and reducing dropout rates.

- **Top features identified:**  
  Past grades (G1, G2), study time, family educational support, and parental involvement are strong predictors, consistent with education research.

---

## 📈 Results & Discussion

- **Best Model Performance (classification, e.g. Random Forest):**
  - Accuracy: ~XX% | F1-score: ~XX% | ROC-AUC: ~XX  *(replace with real results)*
- **Best Regression Model:**
  - R²: ~XX | MAE: ~XX | RMSE: ~XX  *(replace with real results)*
- **Key findings:**  
  - Previous grades and family support are highly predictive.
  - Models generalize well with proper preprocessing and validation.

---

## 🚩 Future Work

- **Modeling:** Experiment with advanced ensembles (e.g. XGBoost, stacking), neural networks
- **Feature Engineering:** Create aggregate variables; test dimensionality reduction
- **Deployment:** Build a web/mobile app for live predictions by educators
- **Generalization:** Integrate data from additional schools or years

---

## 👩‍💻 Contributors

- **Prakash-codeMaker** – Data Scientist / Developer *(update this with your actual name and contributors)*

---

## 📝 License

- Specify the license here (e.g., MIT License).

---

## 📚 References

- Cortez, P., & Silva, A. (2008). [Using Data Mining to Predict Secondary School Student Performance](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- [Frontiers in AI & Education Research](https://www.frontiersin.org/)
- [Project inspirations and similar works on GitHub](https://github.com/)
- Others as cited within notebook/report

---

*Replace all placeholder sections (like video link, XX metrics) with your real values before submission.*

