import streamlit as st
import pickle
import numpy as np

# Load the model and column names
model = pickle.load(open("C:\\Users\\Jhanvi\\Titanic.pkl", 'rb'))
columns = pickle.load(open("C:\\Users\\Jhanvi\\Titanic_cols.pkl", 'rb'))


# Function to take user input and make prediction
def user_input_features():
    pclass = st.selectbox('Pclass', [1, 2, 3])
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.slider('Age', 0, 100, 25)
    fare = st.slider('Fare', 0, 500, 100)
    embarked = st.selectbox('Embarked', ['S', 'Q', 'C'])
    family_size = st.selectbox('Family Size', ['alone', 'small', 'large'])

    # Convert user input to model input format
    data = {
        'Age': age,
        'Fare': fare,
        f'Pclass_{pclass}': 1,
        f'Sex_{sex}': 1,
        f'Embarked_{embarked}': 1,
        f'family_size_{family_size}': 1
    }

    # Initialize all the columns with 0
    user_input = np.zeros(len(columns))

    # Update the user input with the values from the dictionary
    for key, value in data.items():
        if key in columns:
            user_input[columns.index(key)] = value

    return np.array([user_input])


# Streamlit app interface
st.title('Titanic Survival Prediction')
st.write('This app predicts if a passenger survived the Titanic disaster based on their information.')

# Get user input
input_data = user_input_features()

# Make prediction
prediction = model.predict(input_data)

# Display result
st.subheader('Prediction')
if prediction[0] == 1:
    st.success('Survived')
else:
    st.error('Did not survive')
