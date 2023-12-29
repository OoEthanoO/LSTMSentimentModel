import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data import data
import os

app = Flask(__name__)

if os.environ.get('GAE_APPLICATION', None):
    app.config['SQLALCHEMY_DATABASE_URI'] = (
        'postgresql+psycopg2://postgres:pyhPTI{(tn[I~ZnG@/feedback-db'
        '?host=/cloudsql/sentimentstudio-409418:us-central1:feedback-db')
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'

db = SQLAlchemy(app)


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    expected = db.Column(db.String(20), nullable=False)
    displayed = db.Column(db.String(20), nullable=False)


with app.app_context():
    db.create_all()

def handle_negation(text):
    negations = ['not', 'no', 'never', 'none', 'nowhere', 'nothing', 'negative', 'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor', 'dont', 'doesnt',
                  'isnt', 'arent', 'aint', 'nobody', 'darent', 'neednt', 'lack', 'lacks', 'lacking', 'lacked', 'lacking', 'lack', 'lacks', 'lacked', 'lacking', 'lack',
                 'wouldnt', 'shouldnt', 'couldnt', 'wont', 'cant', 'cannot', 'mustnt', 'shant', 'without', 'nevertheless', 'nonetheless', 'except', 'but', 'however',
                 'opposite']
    text = text.lower().replace("'", "").split()
    negation_counter = 0
    for i in range(len(text)):
        if text[i] in negations:
            negation_counter += 1
        if text[i][-1] in '.!?':
            negation_counter = 0
        if negation_counter % 2 != 0:
            text[i] = 'NOT_' + text[i]
    return ' '.join(text)

def get_model_summary():
    from io import StringIO
    import sys

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    model.summary()

    sys.stdout = old_stdout

    return mystdout.getvalue()

def get_and_increment_build_number():
    try:
        with open('build_number.txt', 'r') as f:
            build_number = int(f.read())
    except FileNotFoundError:
        build_number = 0

    with open('build_number.txt', 'w') as f:
        f.write(str(build_number + 1))

    return build_number + 1

model = load_model('Senti2.0.keras')

texts = [handle_negation(text) for text, sentiment in data]
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

reverse_sentiment_dict = {0: "positive", 1: "neutral", 2: "negative"}

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')

#build_number = get_and_increment_build_number()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form['text']
    if user_text == '':
        return jsonify({'sentiment': 'Please enter text', 'probabilities': [0, 0, 0]})
    user_text = handle_negation(user_text)
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

@app.route('/version')
def version():
    model_summary = get_model_summary()
    return render_template('version.html', model_summary=model_summary)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
