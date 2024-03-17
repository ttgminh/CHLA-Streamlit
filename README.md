# CHLA No-Show Predictor

## Project Overview
This project aims to predict the likelihood of no-show appointments at the Children's Hospital Los Angeles (CHLA). Utilizing historical appointment data and machine learning techniques, we've developed a model that assists in identifying patients who might not attend their scheduled appointments. This model is deployed as a web application using Streamlit, offering a user-friendly interface for hospital administrators to input appointment details and receive instant predictions.

## Accessing the Application
You can access the CHLA No-Show Predictor web application through the following link: [CHLA No-Show Predictor App](https://chla-app-6nvmdghbxitxbzvfh89uh7.streamlit.app/).

### Using the Web Application
Upon accessing the link, you will be presented with a simple form where you can input the details of the appointment. These details include:
- Lead Time
- Appointment Type (Follow-up, New, Others)
- Appointment Number
- Total Number of Cancellations
- Total Number of Not Checkout Appointments
- Total Number of Successful Appointments
- Day of the Week
- Age

After filling in the appointment details, click the "Predict" button. The model will then provide a prediction indicating if the patient is likely to show up for their appointment. This tool is designed to help hospital staff manage resources more efficiently and improve patient care by identifying potential no-shows in advance.

## Project Files
The project consists of several key files:
- `app.py`: The main Python script that runs the Streamlit web application.
- `model.pkl`: The trained machine learning model used for making predictions.
- `label_encoder.pkl`: Encodings for categorical variables used by the model.
- `requirements.txt`: A list of Python libraries required to run the application.

## Acknowledgements
This project was developed using data and insights from the Children's Hospital Los Angeles. Special thanks to the CHLA staff and administration for their support and collaboration.
