# LSTMSentimentModel API

The LSTMSentimentModel API is a sentiment analysis service that uses a Long Short-Term Memory (LSTM) model to predict the sentiment of a given text. The API is built with Python and Flask and uses the TensorFlow library for the LSTM model.

## Endpoints

### POST /lstmPredict

This endpoint accepts a JSON object with a single key-value pair. The key should be 'text' and the value should be the text for which you want to predict the sentiment.

#### Request

```json
{
    "text": "your text here"
}
```

#### Response

The response is a JSON object that includes the predicted sentiment and the probabilities for each sentiment category (positive, neutral, negative).

```json
{
    "sentiment": "predicted sentiment",
    "probabilities": [probability for positive, probability for neutral, probability for negative]
}
```

### POST /lstmVersion

This endpoint returns a summary of the LSTM model used for sentiment prediction.

#### Request

No parameters are required for this request.

#### Response

The response is a JSON object that includes a summary of the LSTM model.

```json
{
    "summary": "model summary"
}
```

## Error Handling

If an error occurs while processing the request, the API will return a JSON object with an 'error' key and a description of the error as the value.

```json
{
    "error": "description of the error"
}
```

## Usage

To use this API, you can send a POST request to the appropriate endpoint with the required parameters. The API will return a JSON response with the predicted sentiment or model summary.

## Note

This API is designed to handle negations in the input text. It will prepend 'NOT_' to any word that follows a negation word until the next punctuation mark. This helps the model to better understand the sentiment of the text.
