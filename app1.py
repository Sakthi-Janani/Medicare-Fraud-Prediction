import streamlit as st
import pickle
import numpy as np


# Load Model
try:
    with open("Classifier.pkl", "rb") as pickle_in:
        classifier = pickle.load(pickle_in)
except FileNotFoundError:
    st.error("❌ Error: Model file 'Classifier.pkl' not found.")
    st.stop()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

def predict_fraud(Overcharging_Ratio, Claim_Difference, Payment_ZScore):
    try:
        input_data = np.array([[float(Overcharging_Ratio), float(Claim_Difference), float(Payment_ZScore)]])
        prediction = classifier.predict(input_data)
        return prediction[0]
    except ValueError:
        return "Invalid input"

def main():
    st.title("🩺 Medicare Claims Fraud Detection")

    # User Input
    Overcharging_Ratio = st.number_input("Overcharging Ratio", value=0.0, step=0.01)
    Claim_Difference = st.number_input("Claim Difference", value=0.0, step=0.01)
    Payment_ZScore = st.number_input("Payment Z-Score", value=0.0, step=0.01)

    if st.button("🔍 Predict Fraud"):
        result = predict_fraud(Overcharging_Ratio, Claim_Difference, Payment_ZScore)
        if result == "Invalid input":
            st.error("❌ Please enter valid numbers.")
        else:
            st.success(f"✅ Prediction: **{result}**")

if __name__ == '__main__':
    main()
