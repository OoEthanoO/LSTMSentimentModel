from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from data import data
from main import padded_sequences, tokenizer

sentiments = [sentiment for text, sentiment in data]

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

model.save('Senti2.0.keras')