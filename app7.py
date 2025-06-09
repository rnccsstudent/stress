import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.title("🎓 Student Lifestyle Stress Level Prediction and Analysis")

st.markdown("""
    <style>
    .title span {
        font-size: 24px;
        font-weight: bold;
        animation: colorLoop 8s infinite;
        display: inline-block;
        padding: 0 5px;
    }
    .title span:nth-child(1) { animation-delay: 0s; }
    .title span:nth-child(2) { animation-delay: 1s; }
    .title span:nth-child(3) { animation-delay: 2s; }
    .title span:nth-child(4) { animation-delay: 3s; }
    .title span:nth-child(5) { animation-delay: 4s; }
    .title span:nth-child(6) { animation-delay: 5s; }
    .title span:nth-child(7) { animation-delay: 6s; }
    .title span:nth-child(8) { animation-delay: 7s; }

    @keyframes colorLoop {
        0% { color: red; }
        14% { color: orange; }
        28% { color: green; }
        42% { color: blue; }
        57% { color: indigo; }
        71% { color: violet; }
        85% { color: teal; }
        100% { color: red; }
    }
    </style>

    <div class='title'>
        <span>Student</span>
        <span>Lifestyle</span>
        <span>Stress</span>
        <span>Level</span>
        <span>Prediction</span>
        <span>and</span>
        <span>Analysis</span>
    </div>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("student_lifestyle_dataset.csv")

df = load_data()
st.text("📄 Dataset Preview:")
st.text(df.head().to_string())

# Download dataset button
st.download_button(
    label="📥 Download Dataset (CSV)",
    data=df.to_csv(index=False),
    file_name='student_lifestyle_dataset.csv',
    mime='text/csv'
)

# Preprocess
df = df.drop(columns=["Student_ID"])
label_encoder = LabelEncoder()
df["Stress_Level_Label"] = label_encoder.fit_transform(df["Stress_Level"])

X = df.drop(columns=["Stress_Level", "Stress_Level_Label"])
y = df["Stress_Level_Label"]
class_names = label_encoder.classes_  # ['High', 'Low', 'Moderate']

# User clicks this button to train models and show results
if st.button("▶️ Click me, if you want to see Model Performance and Model Comparison"):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    nb = GaussianNB()
    logreg = LogisticRegression(max_iter=1000)
    svm = SVC(probability=True)

    nb.fit(X_train, y_train)
    logreg.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    # Save logistic regression model only
    joblib.dump(logreg, 'logistic_model.pkl')

    # Evaluate function
    def evaluate_model(model, name):
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            unique_classes = sorted(list(set(y_test)))
            y_bin = label_binarize(y_test, classes=unique_classes)

            if len(unique_classes) == 3:
                auc = roc_auc_score(y_bin, y_proba, multi_class="ovr")
            else:
                auc = float('nan')
                st.warning(f"AUC not calculated for {name} – not all classes present in y_test.")

            return {
                "Model": name,
                "Accuracy": acc,
                "Macro_F1": f1,
                "ROC_AUC_OvR": auc
            }
        except Exception as e:
            st.error(f"❌ Error in {name}: {e}")
            return {
                "Model": name,
                "Accuracy": 0,
                "Macro_F1": 0,
                "ROC_AUC_OvR": 0
            }


    # Calculate metrics
    results = [
        evaluate_model(nb, "Naive Bayes"),
        evaluate_model(logreg, "Logistic Regression"),
        evaluate_model(svm, "SVM")
    ]

    results_df = pd.DataFrame(results)

    st.subheader("📊 Model Performance")

    # Show all models in a table
    st.dataframe(results_df)
    # For Logistic Regression
    #logreg_result = next((r for r in results if r["Model"] == "Logistic Regression"), None)
   # st.text("📄 For Logistic Regression:")
    #if logreg_result:
      # st.write(f"**Accuracy:** {logreg_result['Accuracy']:.2f}")
      # st.write(f"**Macro F1 Score:** {logreg_result['Macro_F1']:.2f}")
      # st.write(f"**ROC AUC Score:** {logreg_result['ROC_AUC_OvR']:.2f}")


    # For SVM
   # svm_result = next((r for r in results if r["Model"] == "SVM"), None)
    #st.text("📄 For SVM:")
    #if svm_result:
       #st.write(f"**Accuracy:** {svm_result['Accuracy']:.2f}")
       #st.write(f"**Macro F1 Score:** {svm_result['Macro_F1']:.2f}")
       #st.write(f"**ROC AUC Score:** {svm_result['ROC_AUC_OvR']:.2f}")

    # For Naive Bayes
   # nb_result = next((r for r in results if r["Model"] == "Naive Bayes"), None)
   # st.text("📄 For Naive Bayes:")
   # if nb_result:
       #st.write(f"**Accuracy:** {nb_result['Accuracy']:.2f}")
       #st.write(f"**Macro F1 Score:** {nb_result['Macro_F1']:.2f}")
       #st.write(f"**ROC AUC Score:** {nb_result['ROC_AUC_OvR']:.2f}")


    # Plot
    try:
        st.subheader("📈 Model Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        results_df.set_index('Model')[['Accuracy', 'Macro_F1', 'ROC_AUC_OvR']].plot(kind='bar', rot=0, ax=ax)
        ax.set_title('Stress Level Prediction Model Comparison')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Plotting failed: {e}")

#else:
     #st.info("▶️ Click the button above to train models and see performance metrics.")

# ------------------------------
# 🧠 User Input Prediction Section
# ------------------------------
st.markdown("""
    <style>
    .header span {
        font-size: 24px;
        font-weight: 600;
        animation: colorLoop 6s infinite;
        display: inline-block;
        padding: 0 5px;
    }
    .header span:nth-child(1) { animation-delay: 0s; }
    .header span:nth-child(2) { animation-delay: 1s; }
    .header span:nth-child(3) { animation-delay: 2s; }
    .header span:nth-child(4) { animation-delay: 3s; }

    @keyframes colorLoop {
        0% { color: #e74c3c; }     /* red */
        25% { color: #f39c12; }    /* orange */
        50% { color: #27ae60; }    /* green */
        75% { color: #2980b9; }    /* blue */
        100% { color: #8e44ad; }   /* purple */
    }
    </style>

    <div class='header'>
        <span>📝 Check</span>
        <span>Your</span>
        <span>Stress</span>
        <span>Level</span>
    </div>
""", unsafe_allow_html=True)


with st.form("stress_form"):
    study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 4.0)
    extra_hours = st.slider("Extracurricular Hours per Day", 0.0, 6.0, 1.0)
    sleep_hours = st.slider("Sleep Hours per Day", 0.0, 12.0, 7.0)
    social_hours = st.slider("Social Hours per Day", 0.0, 8.0, 2.0)
    physical_hours = st.slider("Physical Activity Hours per Day", 0.0, 5.0, 1.0)
    gpa = st.slider("GPA", 0.0, 4.0, 3.0)
    submitted = st.form_submit_button("Predict Stress Level")

if submitted:
    try:
        model = joblib.load("logistic_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model not found! Please train the model first by clicking 'Train and Evaluate Models'.")
    else:
        input_data = pd.DataFrame([[study_hours, extra_hours, sleep_hours, social_hours, physical_hours, gpa]],
                                  columns=X.columns)

        prediction = model.predict(input_data)[0]
        stress_level = label_encoder.inverse_transform([prediction])[0]

        st.success(f"🎯 Predicted Stress Level: **{stress_level}**")

        st.subheader("🧘 Recommendations:")
        if stress_level == "High":
            st.warning("""
            😟 You're showing signs of **High Stress**.
            - Try reducing study overload.
            - Get enough **sleep** and physical exercise.
            - Talk to someone — it's okay to seek help.
            - Use relaxation apps or meditate regularly.
            """)
        elif stress_level == "Moderate":
            st.info("""
            😐 You're in a **Moderate Stress** zone.
            - Try balancing academics and social life.
            - Add short physical breaks while studying.
            - Try planning your time weekly to reduce anxiety.
            """)
        else:
            st.success("""
            😊 You're doing **great with Low Stress**.
            - Keep maintaining your healthy routine.
            - Be consistent with sleep, study, and self-care.
            - Help others manage their stress too!
            """)
        
    # Calculate Study to Sleep Ratio and Academic Intensity
        study_sleep_ratio = round(study_hours / sleep_hours, 2) if sleep_hours > 0 else 0
        academic_intensity = round((study_hours + gpa * 2), 2)

# Calculate Total Active Time
        total_active_time = round(study_hours + physical_hours + social_hours, 2)

# Calculate Stress Risk Score
        stress_risk_score = round(
           (study_sleep_ratio * 0.4 + total_active_time * 0.3 + (4.0 - gpa) * 0.3), 2
             )

# Display the calculated values
        st.subheader("📌 Additional Insights")
        st.write(f"**Study to Sleep Ratio:** {study_sleep_ratio}")
        st.write(f"**Academic Intensity:** {academic_intensity}")
        st.write(f"**Total Active Time (hrs/day):** {total_active_time}")
        st.write(f"**Stress Risk Score (0-10):** {stress_risk_score}")



       

