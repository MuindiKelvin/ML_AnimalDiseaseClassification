import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Loading the dataset 📊
df = pd.read_csv("Data/data.csv", usecols=['AnimalName', 'symptoms1', 'symptoms2', 'symptoms3', 'symptoms4', 'symptoms5', 'Type_of_Disease'])

# Separating features and target variable 🔍
X = df.drop(["AnimalName", "Type_of_Disease"], axis=1)
y = df["Type_of_Disease"]

# Splitting the dataset into training and testing sets 📊
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the pipeline ⚙️
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Handling missing values with the most frequent value 🛠️
    ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Encoding categorical variables 🏷️
])

# Pre-processing the training set 🛠️
X_train_preprocessed = pipeline.fit_transform(X_train)

# Pre-processing the testing set 🛠️
X_test_preprocessed = pipeline.transform(X_test)

# Creating a random forest classifier 🌳
model = RandomForestClassifier(random_state=42)

# Training the model 🚂
model.fit(X_train_preprocessed, y_train)

# Defining improved disease severity classification
def classify_severity(probabilities, threshold_dangerous=0.7, threshold_mild=0.4):
    max_prob = max(probabilities)
    if max_prob >= threshold_dangerous:
        return "⚠️ High Risk", "red"
    elif max_prob >= threshold_mild:
        return "🟡 Moderate Risk", "yellow"
    else:
        return "🟢 Low Risk", "green"

# Defining function to plot feature importance
def plot_feature_importance(model, pipeline, user_input):
    # Getting feature names after one-hot encoding
    feature_names = pipeline.named_steps['encoder'].get_feature_names_out(X.columns)
    
    # Getting feature importances
    importances = model.feature_importances_
    
    # Creating a dictionary to store feature importances
    feature_importance_dict = dict(zip(feature_names, importances))
    
    # Filtering importances based on user input
    filtered_importances = []
    filtered_names = []
    for symptom, value in user_input.items():
        feature_name = f"{symptom}_{value}"
        if feature_name in feature_importance_dict:
            filtered_importances.append(feature_importance_dict[feature_name])
            filtered_names.append(f"{symptom}: {value}")
    
    # Sorting feature importances in descending order
    sorted_idx = np.argsort(filtered_importances)[::-1]  # Reverse order for descending
    sorted_importances = np.array(filtered_importances)[sorted_idx]
    sorted_names = np.array(filtered_names)[sorted_idx]

    # Calculating percentages
    total_importance = sum(sorted_importances)
    percentages = [imp / total_importance * 100 for imp in sorted_importances]

    # Creating a color map
    n_colors = len(sorted_importances)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))

    # Creating plot
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title("Feature Importance of Selected Symptoms")
    bars = ax.bar(range(len(sorted_importances)), sorted_importances, color=colors)
    ax.set_xticks(range(len(sorted_importances)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylabel("Importance")

    # Adding percentage labels on the bars
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{percentage:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=8)

    plt.tight_layout()
    
    # Returning the plot as a Streamlit figure
    return plt

# Defining the Streamlit app 🚀
def main():
    st.set_page_config(page_title="🐈AnimalCare App: Symptom-Based Disease Predictor🐎🩺", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

    # App theme mode selection
    st.sidebar.title("App Theme Settings ⚙️")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Light 🌞", "Dark 🌙"])
    
    # Applying custom CSS for both modes
    st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        padding: 10px 20px !important;
        text-align: center !important;
        text-decoration: none !important;
        display: inline-block !important;
        font-size: 16px !important;
        margin: 4px 2px !important;
        cursor: pointer !important;
        border-radius: 8px !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Applying custom CSS for dark mode
    if app_mode == "Dark 🌙":
        st.markdown("""
        <style>
        .stApp {
            background-color: #837A7A;
            color: #FFFFFF;
        }
        .stSelectbox {
            color: #FFFFFF;
        }
        </style>
        """, unsafe_allow_html=True)

    # Adding a title
    st.title("🐾🐈Symptom-Based Disease Classification🐎🩺")

    # Adding introduction text
    st.write("""
This app predicts whether an animal's condition is dangerous based on its symptoms and characteristics.
📝 Please fill in the information below:
""")

    # Creating three columns
    col1, col2, col3 = st.columns(3)

    # Adding animal selection
    with col1:
        animal = st.selectbox("Select Animal 🐶😺🐴🐄🐖🐔🦜", ['Dog', 'Cat', 'Horse', 'Cow', 'Pig', 'Goat', 'Duck', 'Sheep', 'Guinea Pig', 'Rabbit', 'Fowl', 'Others'])

    # Adding input fields for symptoms 💉
    with col1:
        symptoms1 = st.selectbox("Symptom 1 🤒", X["symptoms1"].unique())
        symptoms2 = st.selectbox("Symptom 2 🤕", X["symptoms2"].unique())

    with col2:
        symptoms3 = st.selectbox("Symptom 3 🤧", X["symptoms3"].unique())
        symptoms4 = st.selectbox("Symptom 4 🤢", X["symptoms4"].unique())

    with col3:
        symptoms5 = st.selectbox("Symptom 5 🤯", X["symptoms5"].unique())

    if st.button("Predict 🧪", key="predict_button"):
        # Checking for duplicate inputs
        user_input = [symptoms1, symptoms2, symptoms3, symptoms4, symptoms5]
        if len(set(user_input)) != len(user_input):
            st.warning("⚠️ You cannot provide similar inputs! Please provide unique inputs for each symptom.")
        else:
            # Creating a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simulating a loading process
            for i in range(100):
                status_text.text(f"Processing: {i+1}%")
                progress_bar.progress(i + 1)
                time.sleep(0.01)

            # Preprocessing user inputs 🛠️
            user_input = pd.DataFrame({
                "symptoms1": [symptoms1],
                "symptoms2": [symptoms2],
                "symptoms3": [symptoms3],
                "symptoms4": [symptoms4],
                "symptoms5": [symptoms5],
            })
            user_input_preprocessed = pipeline.transform(user_input)

            # Making predictions 📈
            prediction = model.predict(user_input_preprocessed)
            probabilities = model.predict_proba(user_input_preprocessed)[0]

            # Removing the progress bar and status text
            progress_bar.empty()
            status_text.empty()

            # Classifying the severity
            severity, color = classify_severity(probabilities)

            # Displaying the predicted disease with severity
            st.write(f"The {animal} is predicted to have **{prediction[0]}** disease.")
            st.markdown(f"<span style='color:{color};font-weight:bold;'>Severity: {severity}</span>", unsafe_allow_html=True)
            
            # Displaying prediction probabilities
            st.write("### Prediction Probabilities 📊:")
            for disease, prob in zip(model.classes_, probabilities):
                percentage = prob * 100
                st.markdown(f"**{disease}: {percentage:.2f}%**")

            # Displaying feature importance plot
            st.write("### Feature Importance of Selected Symptoms 📊:")
            fig = plot_feature_importance(model, pipeline, user_input.iloc[0])
            st.pyplot(fig)

# Running the app 🏃
if __name__ == "__main__":
    main()

    st.markdown("""
    **Note: This project is developed using a fictional dataset from Kaggle.com. Always consult with a qualified veterinarian for proper diagnosis and treatment of animal diseases. 🐾**
    """)