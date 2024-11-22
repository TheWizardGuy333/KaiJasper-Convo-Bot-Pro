from textblob import TextBlob
import pandas as pd
from pathlib import Path

# Define the path for storing feedback data
FEEDBACK_FILE = Path("data/feedback_data.csv")

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text.

    Args:
        text (str): Input text for sentiment analysis.

    Returns:
        dict: Polarity and subjectivity of the text.
    """
    blob = TextBlob(text)
    return {"polarity": blob.polarity, "subjectivity": blob.subjectivity}

def learn_from_feedback():
    """
    Placeholder for learning from user feedback. 
    Currently, it checks the feedback file and provides a message.

    Returns:
        str: A message about learning progress or data availability.
    """
    if not FEEDBACK_FILE.exists():
        return "No feedback data to train on."
    
    feedback_data = pd.read_csv(FEEDBACK_FILE)
    # Placeholder logic for using feedback to improve sentiment models
    return "Feedback data used to improve learning pipeline."

def save_feedback(text, sentiment, feedback):
    """
    Saves user feedback about sentiment analysis results.

    Args:
        text (str): Input text analyzed for sentiment.
        sentiment (dict): Sentiment analysis results.
        feedback (str): User feedback (e.g., "Yes" or "No").
    """
    data = {
        "text": [text],
        "polarity": [sentiment["polarity"]],
        "subjectivity": [sentiment["subjectivity"]],
        "feedback": [feedback]
    }
    df = pd.DataFrame(data)
    
    # Create the file if it doesn't exist; append if it does
    if FEEDBACK_FILE.exists():
        df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
    else:
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(FEEDBACK_FILE, index=False)
