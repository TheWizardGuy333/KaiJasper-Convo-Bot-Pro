import pandas as pd
from pathlib import Path

# File paths for storing data
CHAT_HISTORY_FILE = Path("data/chat_history.csv")
FEEDBACK_FILE = Path("data/feedback_data.csv")

def save_chat(text, sentiment):
    """
    Save chatbot interaction to the chat history file.

    Args:
        text (str): The user input text.
        sentiment (dict): Sentiment analysis results containing polarity and subjectivity.

    Returns:
        str: Success message upon saving the chat.
    """
    data = {
        "text": [text],
        "polarity": [sentiment["polarity"]],
        "subjectivity": [sentiment["subjectivity"]],
    }
    df = pd.DataFrame(data)

    try:
        if CHAT_HISTORY_FILE.exists():
            df.to_csv(CHAT_HISTORY_FILE, mode="a", header=False, index=False)
        else:
            CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(CHAT_HISTORY_FILE, index=False)

        return "Chat history saved successfully."
    except Exception as e:
        return f"Error saving chat history: {e}"

def get_chat_history():
    """
    Retrieve the chat history from the saved file.

    Returns:
        DataFrame or str: Chat history as a DataFrame if available, else an error message.
    """
    try:
        if CHAT_HISTORY_FILE.exists():
            return pd.read_csv(CHAT_HISTORY_FILE)
        else:
            return "No chat history found."
    except Exception as e:
        return f"Error reading chat history: {e}"

def save_feed(text, feedback):
    """
    Save user feedback to the feedback file.

    Args:
        text (str): The analyzed text.
        feedback (str): User feedback about the response accuracy.

    Returns:
        str: Success message upon saving feedback.
    """
    if not isinstance(feedback, str) or feedback.lower() not in ["yes", "no"]:
        return "Feedback must be 'Yes' or 'No'."

    data = {
        "text": [text],
        "feedback": [feedback.lower()],
    }
    df = pd.DataFrame(data)

    try:
        if FEEDBACK_FILE.exists():
            df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
        else:
            FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(FEEDBACK_FILE, index=False)

        return "Feedback saved successfully."
    except Exception as e:
        return f"Error saving feedback: {e}"
