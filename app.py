import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ضبط إعدادات الصفحة
st.set_page_config(
    page_title="Student Risk Category Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 نظام التنبؤ بمستوى خطورة الطالب (Student Risk Category)")
st.write(
    "يقوم هذا التطبيق بتحليل بيانات الطالب وتوقع فئة الخطورة الخاصة به بناءً على نموذج الذكاء الاصطناعي."
)


# 1. تحميل البيانات وتجهيزها
@st.cache_data
def load_and_preprocess_data():
    # ملاحظة: قم بتعديل مسار الملف بحسب موقع الملف لديك
    df = pd.read_csv("student_data.csv")

    # معالجة Outliers بالـ clipping كما في النوت بوك
    numeric_cols = df.select_dtypes(include="number").columns
    outlier_cols = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        if len(df[(df[col] < lower) | (df[col] > upper)]) > 0:
            outlier_cols.append(col)
            df[col] = df[col].clip(lower=lower, upper=upper)

    X = df.drop("risk_category", axis=1)
    y = df["risk_category"]

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # معالجة القيم المفقودة (Imputation)
    num_cols = X_train.select_dtypes(include="number").columns
    imputer = SimpleImputer(strategy="median")
    X_train[num_cols] = imputer.fit_transform(X_train[num_cols])

    # تحويل المتغيرات النصية (One-Hot Encoding)
    cat_cols = X_train.select_dtypes(include="object").columns
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols)

    # تشفير المتغير التابع (Target Label Encoding)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # القياس المعياري (Standard Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)

    # تدريب نموذج Random Forest كنموذج ممتاز
    model = RandomForestClassifier(max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train_encoded)

    return (
        df,
        imputer,
        scaler,
        le,
        model,
        X_train_encoded.columns,
        num_cols,
        cat_cols,
    )


# تحميل البيانات والنموذج
(
    df,
    imputer,
    scaler,
    le,
    model,
    feature_columns,
    num_cols,
    cat_cols,
) = load_and_preprocess_data()

# 2. واجهة إدخال البيانات في الشريط الجانبي (Sidebar)
st.sidebar.header("📋 أدخل بيانات الطالب:")

attendance = st.sidebar.slider(
    "نسبة الحضور (Attendance %)", 0.0, 100.0, 75.0, step=0.1
)
study_hours = st.sidebar.slider("ساعات الدراسة (Study Hours)", 0.0, 15.0, 6.0, step=0.1)
past_failures = st.sidebar.number_input(
    "عدد مرات الرسوب السابقة (Past Failures)",
    min_value=0,
    max_value=5,
    value=1,
)
assignments_completed_pct = st.sidebar.slider(
    "نسبة الواجبات المكتملة (Assignments Completed %)", 0.0, 100.0, 70.0, step=0.1
)
previous_grade = st.sidebar.slider("الدرجة السابقة (Previous Grade)", 0.0, 100.0, 65.0, step=0.1)
final_score = st.sidebar.slider("الدرجة النهائية (Final Score)", 0.0, 100.0, 60.0, step=0.1)

parental_education = st.sidebar.selectbox(
    "مستوى تعليم الوالدين", ["Primary", "Secondary", "Higher"]
)
family_income = st.sidebar.selectbox("مستوى دخل الأسرة", ["Low", "Medium", "High"])
extracurricular = st.sidebar.selectbox("الأنشطة اللاصفية", ["Yes", "No"])
internet_access = st.sidebar.selectbox("توفر الإنترنيت", ["Yes", "No"])

# 3. عرض البيانات المدخلة وتجهيزها للتوقع
st.subheader("📊 البيانات المدخلة للطالب:")
input_dict = {
    "attendance": attendance,
    "study_hours": study_hours,
    "past_failures": past_failures,
    "assignments_completed_pct": assignments_completed_pct,
    "parental_education": parental_education,
    "family_income": family_income,
    "extracurricular": extracurricular,
    "internet_access": internet_access,
    "previous_grade": previous_grade,
    "final_score": final_score,
}

input_df = pd.DataFrame([input_dict])
st.dataframe(input_df)

# زر التوقع
if st.button("🔮 تنبؤ بفئة الخطورة"):
    # معالجة المدخلات بنفس الخطوات
    input_df[num_cols] = imputer.transform(input_df[num_cols])
    input_df_encoded = pd.get_dummies(input_df, columns=cat_cols)

    # مطابقة الأعمدة مع أعمدة التدريب (Reindex)
    input_df_encoded = input_df_encoded.reindex(
        columns=feature_columns, fill_value=0
    )

    # القياس المعياري
    input_scaled = scaler.transform(input_df_encoded)

    # التوقع
    prediction_encoded = model.predict(input_scaled)
    prediction_label = le.inverse_transform(prediction_encoded)[0]

    # عرض النتيجة
    st.markdown("---")
    st.subheader("🎯 النتيجة:")

    if prediction_label == "Safe":
        st.success(f"فئة الطالب: **{prediction_label}** 🟢 (الطالب في وضع آمن)")
    elif prediction_label == "At-Risk":
        st.warning(
            f"فئة الطالب: **{prediction_label}** 🟡 (الطالب معرض للتردي الأكاديمي)"
        )
    else:
        st.error(
            f"فئة الطالب: **{prediction_label}** 🔴 (الطالب في مرحلة خطورة عالية)"
        )