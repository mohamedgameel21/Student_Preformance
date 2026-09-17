Markdown
# 🎓 Student Performance & Risk Category Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)

An end-to-end Machine Learning project and interactive web application designed to analyze academic and demographic student data to predict a student's **Risk Category**. This tool helps educational institutions identify students who need early academic intervention.

---

## 📌 Features

* **Data Preprocessing Pipeline:**
  * Handles missing values using Median Imputation.
  * Manages numerical outliers using IQR clipping techniques.
  * Encodes categorical variables via One-Hot Encoding and target variable via Label Encoding.
  * Scales numerical features using `StandardScaler`.
* **Machine Learning Model:**
  * Trained using a **Random Forest Classifier** tuned for high prediction accuracy and stability.
* **Interactive Streamlit Dashboard:**
  * User-friendly sidebar interface to input custom student metrics and generate real-time predictions.

---

## 🛠️ Local Installation & Setup

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/mohamedgameel21/Student_Preformance.git](https://github.com/mohamedgameel21/Student_Preformance.git)
cd Student_Preformance
2. Install Dependencies
Ensure you have Python installed, then install the required packages:

Bash
pip install pandas numpy scikit-learn streamlit
3. Run the Application
Start the Streamlit server:

Bash
streamlit run app.py
📊 Dataset Features & Inputs
The prediction model evaluates both academic metrics and socio-economic factors:
```
Academic Metrics: Attendance %, Study Hours, Past Failures, Assignments Completed %, Previous Grade, and Final Score.

Demographic Factors: Parental Education Level, Family Income, Extracurricular Activities, and Internet Access.

🎯 Prediction Categories
The model categorizes students into one of three risk levels:

🟢 Safe: The student is performing well and academically secure.

🟡 At-Risk: The student shows signs of academic decline and requires monitoring.

🔴 High Risk: The student is at severe risk of failure and needs immediate intervention.

👤 Author
Developed and maintained by Mohamed Gameel.
