import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import os
from data import data
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import numpy as np

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


def pad_sequences(sequences, padding='post'):
    # Get the length of the longest sequence
    max_len = max(len(seq) for seq in sequences)

    # Create a new list of sequences
    padded_sequences = []

    for seq in sequences:
        # Get the number of zeros to pad
        num_padding = max_len - len(seq)

        if padding == 'post':
            # Pad the sequence with zeros at the end
            padded_seq = seq + [0] * num_padding
        else:
            # Pad the sequence with zeros at the beginning
            padded_seq = [0] * num_padding + seq

        # Add the padded sequence to the new list
        padded_sequences.append(padded_seq)

    return padded_sequences

class SentimentDataset(Dataset):
    def __init__(self, sequences, sentiments):
        self.sequences = sequences
        self.sentiments = sentiments

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.sentiments[idx])

texts = [text for text, sentiment in data]
sentiments = [sentiment for text, sentiment in data]

sentiment_dict = {'positive': 0, 'neutral': 1, 'negative': 2}

tokenizer = get_tokenizer('basic_english')
tokenized_texts = [tokenizer(text) for text in texts]
vocab = build_vocab_from_iterator(tokenized_texts)

UNK_TOKEN = "<unk>"

if UNK_TOKEN not in vocab:
    vocab.append_token(UNK_TOKEN)

vocab.set_default_index(vocab[UNK_TOKEN])
sequences = [[vocab[token] for token in text] for text in tokenized_texts]
padded_sequences = pad_sequences(sequences, padding='post')

sentiments_numerical = [sentiment_dict[sentiment] for sentiment in sentiments]
sentiments_one_hot = np.eye(len(sentiment_dict))[sentiments_numerical]

reverse_sentiment_dict = {i: sentiment for sentiment, i in sentiment_dict.items()}

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, sentiments_one_hot, test_size=0.2,
                                                    random_state=42)

train_data = SentimentDataset(X_train, y_train)
test_data = SentimentDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = self.fc(lstm_out[:, -1, :])
        return self.softmax(out)

model = SentimentModel(len(vocab) + 2, 3, 128, 128, 1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_text = request.form['text']
    if user_text == '':
        return jsonify({'sentiment': 'Please enter text', 'probabilities': [0, 0, 0]})
    user_sequence = [vocab[token] for token in tokenizer(user_text)]
    user_padded_sequence = pad_sequences([user_sequence], padding='post')
    user_tensor = torch.tensor(user_padded_sequence)
    user_prediction = model(user_tensor).detach().numpy()[0]
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
