import numpy as np  
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# load the pre-trained model 
model = load_model('simpleRNN.h5')

#step 2: helper functions
# function to decode review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# function to preprocess user input
def preprocess_text(text):
    words= text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 is for unknown words
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

#step 3: prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>=0.5 else 'Negative'
    return sentiment, prediction[0][0]


import streamlit as st

# Streamlit web app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive/Negative).")

# User input
user_input = st.text_area("Movie Review")

if st.button("classify"):
    preprocessed_input=preprocess_text(user_input)

    #make predictions
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>=0.5 else 'Negative'

    ##display results
    st.write(f"Predicted Sentiment: **{sentiment}**")
    st.write(f"Prediction Confidence: **{prediction[0][0]:.4f}**")
else:
    st.write("Please enter a movie review and click 'Classify' to see the sentiment prediction.")