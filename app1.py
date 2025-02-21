import streamlit as st
import pickle

pickle_in = open("Classifier.pkl","rb")
classifier = pickle.load(pickle_in)

def predict_note_authentication(Overcharging_Ratio,Claim_Difference,Payment_ZScore):

    Overcharging_Ratio = float(Overcharging_Ratio)
    Claim_Difference = float(Claim_Difference)
    Payment_ZScore = float(Payment_ZScore)

    prediction=classifier.predict([[Overcharging_Ratio,Claim_Difference,Payment_ZScore]])
    return prediction(0)

def main():
    st.title("Fraud Detection in Medicare Claims")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)

    Overcharging_Ratio = st.text_input("Overcharging_Ratio","0.0")
    Claim_Difference = st.text_input("Claim_Difference","0.0")
    Payment_ZScore  = st.text_input("Payment_ZScore","0.0")
    result=""

    if st.button("Predict"):
       try:
            result = predict_note_authentication(Overcharging_Ratio, Claim_Difference, Payment_ZScore)
            st.success(f'The prediction result is: {result}')
       except ValueError:
            st.error("Please enter valid numeric values.")


    if __name__=='__main__':
        main()
    