import numpy as np
import json
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

    lstmModel.summary()

    sys.stdout = old_stdout

    return mystdout.getvalue()

app = Flask(__name__)

lstmModel = load_model('LSTMSenti2.1Class_Weighted.keras')

data = []
for filename in ['AMAZON_FASHION_5.json', 'All_beauty_5.json']:
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))

texts = [handle_negation(review['reviewText']) for review in data if 'reviewText' in review]
lstmTokenizer = Tokenizer()
lstmTokenizer.fit_on_texts(texts)

reverse_sentiment_dict = {0: "positive", 1: "neutral", 2: "negative"}

sequences = lstmTokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')

@app.route('/lstmPredict', methods=['POST'])
def lstmPredict():
    data = request.get_json(force=True)
    user_text = data['text']
    print("LSTM: " + user_text)
    user_text = handle_negation(user_text)
    user_sequence = lstmTokenizer.texts_to_sequences([user_text])
    if not user_sequence[0]:
        return jsonify({'sentiment': 'Please enter text', 'probabilities': [0, 0, 0]})
    user_padded_sequence = pad_sequences(user_sequence, padding='post')
    user_prediction = lstmModel.predict(user_padded_sequence)[0]
    max_index = np.argmax(user_prediction)
    predicted_sentiment = reverse_sentiment_dict[max_index]
    return jsonify({'sentiment': predicted_sentiment, 'probabilities': user_prediction.tolist()})

@app.route('/lstmVersion', methods=['POST'])
def lstmVersion():
    model_summary = get_model_summary()
    return jsonify({'summary': model_summary})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)