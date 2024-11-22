import pandas as pd
import matplotlib.pyplot as plt

def plot_sentiment_distribution():
    df = pd.read_csv("data/chat_history.csv")
    if df.empty:
        raise FileNotFoundError("No data available.")
    plt.figure(figsize=(10, 6))
    plt.hist(df["polarity"], bins=20, alpha=0.7, color="blue", edgecolor="black")
    plt.title("Sentiment Polarity Distribution")
    plt.xlabel("Polarity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
