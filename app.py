import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the predictor model from a pickle file
model = pickle.load(open('logistic_model_coefficients.pkl', 'rb'))

# Load the encoder dictionary from a pickle file
with open('encoder.pkl', 'rb') as pkl_file:
    encoder_dict = pickle.load(pkl_file)

def encode_features(df, encoder_dict):
    # For each categorical feature, apply the encoding
    category_col = ['APPT_TYPE_STANDARDIZE']
    for col in category_col:
        if col in encoder_dict:
            le = LabelEncoder()
            le.classes_ = np.array(encoder_dict[col], dtype=object)  # Load the encoder classes for this column

            # Handle unknown categories by using 'transform' method and a lambda function
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            df[col] = le.transform(df[col])
    return df

def main():
    st.title("CHLA Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">CHLA No Show Predictor App </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html = True)

    LEAD_TIME = st.text_input("LEAD_TIME","0")
    APPT_TYPE_STANDARDIZE = st.selectbox("APPT_TYPE_STANDARDIZE", ["Follow-up","New","Others"])
    APPT_NUM= st.text_input("APPT_NUM","0")
    TOTAL_NUMBER_OF_CANCELLATIONS = st.text_input("TOTAL_NUMBER_OF_CANCELLATIONS","0")
    TOTAL_NUMBER_OF_NOT_CHECKOUT_APPOINTMENT= st.text_input("TOTAL_NUMBER_OF_NOT_CHECKOUT_APPOINTMENT","0")
    TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT= st.text_input("TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT","0")
    DAY_OF_WEEK = st.selectbox("DAY_OF_WEEK","0")

    if st.button("Predict"):
        data = {'LEAD_TIME': int(LEAD_TIME), 'APPT_TYPE_STANDARDIZE': APPT_TYPE_STANDARDIZE, 'APPT_NUM': int(APPT_NUM), 'TOTAL_NUMBER_OF_CANCELLATIONS': int(TOTAL_NUMBER_OF_CANCELLATIONS), 'TOTAL_NUMBER_OF_NOT_CHECKOUT_APPOINTMENT': int(TOTAL_NUMBER_OF_NOT_CHECKOUT_APPOINTMENT), 'TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT': int(TOTAL_NUMBER_OF_SUCCESS_APPOINTMENT), 'DAY_OF_WEEK': int(DAY_OF_WEEK)}
        # Convert the data into a DataFrame for easier manipulation
        df = pd.DataFrame([data])

        # Encode the categorical columns
        df = encode_features(df, encoder_dict)

        # Now, all your features should be numerical, and you can attempt prediction
        features_list = df.values

        # Make prediction
        prediction = model.predict(features_list)
        if prediction[0] == 1:
            st.success("This patient is likely to show up for the appointment")
        else:
            st.error("This patient is likely to not show up for the appointment")

if __name__ == '__main__':
    main()
