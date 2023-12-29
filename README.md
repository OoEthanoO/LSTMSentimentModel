# Sentiment Analysis Web Application

This project is a sentiment analysis web application built with Python, Flask, and TensorFlow. It uses a pre-trained model to predict the sentiment (positive, neutral, or negative) of user-provided text. The application also handles negations in the text to improve the accuracy of the sentiment prediction.

## Features

- **Sentiment Prediction**: The application uses a pre-trained TensorFlow model to predict the sentiment of user-provided text.
- **Negation Handling**: The application handles negations in the text to improve the accuracy of the sentiment prediction.
- **Web Interface**: The application provides a user-friendly web interface for users to input text and view the predicted sentiment.
- **Feedback System**: Users can provide feedback on the predicted sentiment. This feedback is stored in a database and can be viewed on a separate page of the web application.
- **Google Cloud App Engine Deployment**: The application is designed to be deployed on Google Cloud App Engine.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/OoEthanoO/SentimentAnalysisWebApp.git
```
2. Navigate to the project directory:
```bash
cd SentimentAnalysisWebApp
```
3. Install the required Python packages:
```bash
pip install -r requirements.txt
```
4. Run the application:
```bash
python main.py
```

## Usage

Open your web browser and navigate to `http://localhost:3000`. Enter your text in the input field and click the "Predict" button to get the sentiment prediction.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Feedbacks

Feedbacks can be submitted on the website. 
