from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import os
from data import data

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'test.db')
db = SQLAlchemy(app)


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    expected = db.Column(db.String(20), nullable=False)
    displayed = db.Column(db.String(20), nullable=False)


with app.app_context():
    db.create_all()

texts = [text for text, sentiment in data]
sentiments = [sentiment for text, sentiment in data]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')

sentiment_dict = {"positive": 0, "neutral": 1, "negative": 2}
sentiments_numerical = [sentiment_dict[sentiment] for sentiment in sentiments]
sentiments_one_hot = to_categorical(sentiments_numerical)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, sentiments_one_hot, test_size=0.2,
                                                    random_state=42)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

reverse_sentiment_dict = {0: "positive", 1: "neutral", 2: "negative"}


# user_text = input("Enter text: ")
# user_sequence = tokenizer.texts_to_sequences([user_text])
# user_padded_sequence = pad_sequences(user_sequence, padding='post')
# user_prediction = model.predict(user_padded_sequence)
# print(user_prediction)
# max_index = np.argmax(user_prediction)
# predicted_sentiment = reverse_sentiment_dict[max_index]
# print(predicted_sentiment)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form['text']
    if user_text == '':
        return jsonify({'sentiment': 'Please enter text', 'probabilities': [0, 0, 0]})
    user_sequence = tokenizer.texts_to_sequences([user_text])
    user_padded_sequence = pad_sequences(user_sequence, padding='post')
    user_prediction = model.predict(user_padded_sequence)[0]
    max_index = np.argmax(user_prediction)
    predicted_sentiment = reverse_sentiment_dict[max_index]
    return jsonify({'sentiment': predicted_sentiment, 'probabilities': user_prediction.tolist()})


@app.route('/feedback', methods=['POST'])
def feedback():
    feedback = Feedback(text=request.form['text'], expected=request.form['expected'],
                        displayed=request.form['displayed'])
    db.session.add(feedback)
    db.session.commit()
    return jsonify({'message': 'Feedback submitted'})


@app.route('/feedbacks')
def feedbacks():
    feedbacks = Feedback.query.all()
    return render_template('feedbacks.html', feedbacks=feedbacks)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
