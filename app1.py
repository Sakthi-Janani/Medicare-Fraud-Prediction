import streamlit as st  # Import Streamlit first

# Set page config at the very top
st.set_page_config(page_title="Medicare Fraud Detection", page_icon="🩺", layout="wide")

import pickle
import numpy as np
import pandas as pd

# Cache model loading for performance
@st.cache_resource
def load_model():
    try:
        with open("Classifier.pkl", "rb") as pickle_in:
            return pickle.load(pickle_in)
    except FileNotFoundError:
        st.error("❌ Error: Model file 'Classifier.pkl' not found.")
        st.stop()

classifier = load_model()

def predict_fraud(Overcharging_Ratio, Claim_Difference, Payment_ZScore):
    try:
        input_data = np.array([[float(Overcharging_Ratio), float(Claim_Difference), float(Payment_ZScore)]])
        prediction = classifier.predict(input_data)
        return prediction[0]
    except ValueError:
        return "Invalid input"

def batch_predict(file):
    df = pd.read_csv(file)
    required_columns = ['Overcharging_Ratio', 'Claim_Difference', 'Payment_ZScore']
    
    if not all(col in df.columns for col in required_columns):
        st.error("❌ Invalid CSV format! Make sure it contains Overcharging_Ratio, Claim_Difference, and Payment_ZScore.")
        return None

    predictions = classifier.predict(df[required_columns])
    df['Fraud_Prediction'] = predictions
    return df

def main():
    st.title("🩺 Medicare Claims Fraud Detection")
    st.write("Use this tool to predict potential fraud in Medicare claims based on key financial indicators.")

    tab1, tab2 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction"])

    with tab1:
        st.subheader("🔍 Single Claim Fraud Check")

        # Sidebar for input
        with st.sidebar:
            st.header("📊 Input Claim Details")
            Overcharging_Ratio = st.number_input("Overcharging Ratio", value=0.0, step=0.01, help="Ratio of billed vs actual cost.")
            Claim_Difference = st.number_input("Claim Difference", value=0.0, step=0.01, help="Difference in expected vs claimed amount.")
            Payment_ZScore = st.number_input("Payment Z-Score", value=0.0, step=0.01, help="Z-Score for payment distribution.")

        if st.button("🔍 Predict Fraud"):
            with st.spinner("Analyzing claim..."):
                result = predict_fraud(Overcharging_Ratio, Claim_Difference, Payment_ZScore)
            
            if result == "Invalid input":
                st.error("❌ Please enter valid numbers.")
            else:
                if result == 1:
                    st.error("⚠️ **Fraud Detected!** This claim is flagged as potentially fraudulent.")
                else:
                    st.success("✅ **No Fraud Detected!** This claim appears legitimate.")

    with tab2:
        st.subheader("📂 Batch Processing (Upload CSV)")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file:
            df_result = batch_predict(uploaded_file)
            if df_result is not None:
                st.write(df_result)
                st.download_button("📥 Download Results", df_result.to_csv(index=False), "fraud_predictions.csv", "text/csv")

if __name__ == '__main__':
    main()
