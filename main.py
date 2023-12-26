from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

data = [("I love this!", "positive"),
        ("I hate this!", "negative"),
        ("This is okay.", "neutral"),
        ("I absolutely loved the new restaurant downtown! The food was delicious, the ambiance was perfect, and the service was exceptional.", "positive"),
        ("I had a terrible experience with the customer service hotline. The representatives were unhelpful, and the wait time was incredibly long.", "negative"),
        ("Today's weather forecast predicts partly cloudy skies with a chance of rain in the afternoon. Temperatures are expected to be in the mid-70s.", "neutral"),
        ("I was disappointed with the quality of the product. It didn't live up to the advertised features, and it broke within a week of use.", "negative"),
        ("The meeting is scheduled for 3 PM tomorrow in the conference room. Please make sure to bring your reports and be prepared for discussions.", "neutral"),
        ("Attending the concert last night was an incredible experience! The music was phenomenal, and the energy in the crowd was contagious.", "positive"),
        ("I was dissatisfied with the accommodations at the hotel. The room wasn't clean, and the amenities were lacking.", "negative"),
        ("The support I received from the tech team was outstanding! They resolved my issue quickly and efficiently.", "positive"),
        ("The traffic this morning was horrendous. I was stuck for over an hour, and it made me late for an important meeting.", "negative"),
        ("The restaurant experience was disappointing. The food was cold, and the service was incredibly slow.", "negative"),
        ("The new gym hours will be effective starting next week. Please check the updated schedule for your preferred workout times.", "neutral"),
        ("The new café in town serves the best coffee I've ever had! The atmosphere is cozy, and the baristas are friendly.", "positive"),
        ("The city marathon will take place next Sunday. Road closures will begin at 7 AM, so plan your travel accordingly.", "neutral"),
        ("Today's weather is looking really good", "positive"),
        ("The concert venue had poor acoustics, and it was hard to enjoy the music with all the sound distortion.", "negative"),
        ("The quarterly financial report will be presented during the board meeting on Friday. Attendance is mandatory for all department heads.", "neutral"),
        ("The city's annual food festival will feature vendors from various cuisines. Don't miss the live cooking demonstrations!", "neutral"),
        ("The yoga retreat was incredibly rejuvenating! The serene environment and skilled instructors made it a memorable experience.", "positive"),
        ("The online course I took was incredibly informative! The instructor was engaging, and the content was well-structured.", "positive"),
        ("The book I read recently was captivating! The characters were well-developed, and the plot twists kept me hooked until the last page.", "positive"),
        ("The new fitness app has revolutionized my workout routine! It's user-friendly and packed with great workout plans.", "positive"),
        ("The concert last night was incredible! The band's performance was energetic, and the crowd was enthusiastic.", "positive"),
        ("The hiking trip was breathtaking! The scenery was stunning, and the fresh air was invigorating.", "positive"),
        ("The airline lost my luggage, and it took days to recover it. Ruined the entire travel experience.", "negative"),
        ("The library will host a book reading session for kids next Saturday. Parents are welcome to bring their children.", "neutral"),
        ("The hotel stay was disappointing. The room was not as clean as expected, and the staff was unresponsive to complaints.", "negative"),
        ("The customer service at the store was exceptional! The staff went above and beyond to assist with my inquiries.", "positive"),
        ("The online delivery was delayed multiple times, causing inconvenience. The lack of communication was frustrating.", "negative"),
        ("The community cleanup day is set for Saturday morning. Volunteers can gather at the community center at 8 AM.", "neutral"),
        ("The hotel stay was wonderful! The room was clean, and the staff was attentive and friendly.", "positive"),
        ("The local festival will feature live music and food stalls. Entry passes can be purchased online or at the gate.", "neutral"),
        ("The concert last night was phenomenal! The musicians' performances were outstanding, and the crowd was ecstatic.", "positive"),
        ("The school's science fair is scheduled for next month. Students can register their projects by the end of the week.", "neutral"),
        ("The school's science fair is scheduled for next month. Students can register their projects by the end of the week.", "neutral"),
        ("The product packaging was damaged upon arrival, and the replacement process was complicated.", "negative"),
        ("The customer support was unresponsive and unhelpful. It was frustrating trying to resolve the issue.", "negative"),
        ("The city council will discuss new traffic regulations at the next session. Residents can attend to understand the changes.", "neutral"),
        ("The food festival was amazing! The variety of cuisines and flavors were a treat for the taste buds.", "positive"),
        ("The food festival was amazing! The variety of cuisines and flavors were a treat for the taste buds.", "positive"),
        ("The customer support team was extremely helpful in resolving my issue. Their assistance was prompt and effective.", "positive"),
        ("I'm drinking water.", "neutral"),
        ("I hate waiting in long lines at the DMV.", "negative"),
        ("I hate dealing with rude customer service.", "negative"),
        ("I detest the taste of this medicine.", "negative"),
        ("The library offers a wide range of books for readers.", "neutral")]

texts = [text for text, sentiment in data]
sentiments = [sentiment for text, sentiment in data]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')

sentiment_dict = {"positive": 0, "neutral": 1, "negative": 2}
sentiments_numerical = [sentiment_dict[sentiment] for sentiment in sentiments]
sentiments_one_hot = to_categorical(sentiments_numerical)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, sentiments_one_hot, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128))
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
        return jsonify({'sentiment': 'Please enter text'})
    user_sequence = tokenizer.texts_to_sequences([user_text])
    user_padded_sequence = pad_sequences(user_sequence, padding='post')
    user_prediction = model.predict(user_padded_sequence)
    max_index = np.argmax(user_prediction)
    predicted_sentiment = reverse_sentiment_dict[max_index]
    return jsonify({'sentiment': predicted_sentiment})

if __name__ == "__main__":
    app.run(host='sentimentstudio.tech', port=8080)