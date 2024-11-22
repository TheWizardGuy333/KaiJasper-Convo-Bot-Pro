import pandas as pd
from pathlib import Path

CHAT_HISTORY_FILE = Path("data/chat_history.csv")

def save_chat(text, sentiment):
    data = {"text": [text], "polarity": [sentiment["polarity"]]}
    df = pd.DataFrame(data)
    CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CHAT_HISTORY_FILE.exists():
        df.to_csv(CHAT_HISTORY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(CHAT_HISTORY_FILE, index=False)

def get_chat_history():
    if CHAT_HISTORY_FILE.exists():
        return pd.read_csv(CHAT_HISTORY_FILE)
    return "No chat history found."
