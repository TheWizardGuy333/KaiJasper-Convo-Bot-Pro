from transformers import pipeline

def sentiment_with_ml(text):
    sentiment_analyzer = pipeline("sentiment-analysis")
    return sentiment_analyzer(text)
