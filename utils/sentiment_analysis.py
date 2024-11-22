from textblob import TextBlob
import pandas as pd
from pathlib import Path
from transformers import pipeline, TrainingArguments, Trainer
from datasets import Dataset

# Path for feedback data storage
FEEDBACK_FILE = Path("data/feedback_data.csv")

def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text using TextBlob.

    Args:
        text (str): Input text for sentiment analysis.

    Returns:
        dict: A dictionary containing polarity and subjectivity scores.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")
    
    blob = TextBlob(text)
    return {
        "polarity": round(blob.polarity, 2),
        "subjectivity": round(blob.subjectivity, 2)
    }

def save_feedback(text, sentiment, feedback):
    """
    Save user feedback for sentiment analysis results.

    Args:
        text (str): The input text analyzed for sentiment.
        sentiment (dict): Sentiment analysis results containing polarity and subjectivity.
        feedback (str): User feedback about the analysis result ("Yes" or "No").

    Returns:
        str: A message indicating the success or failure of the operation.
    """
    if not isinstance(feedback, str) or feedback.lower() not in ["yes", "no"]:
        raise ValueError("Feedback must be either 'Yes' or 'No'.")
    
    data = {
        "text": [text],
        "polarity": [sentiment["polarity"]],
        "subjectivity": [sentiment["subjectivity"]],
        "feedback": [feedback.lower()]
    }
    df = pd.DataFrame(data)

    try:
        # Create the file if it doesn't exist, or append to it
        if FEEDBACK_FILE.exists():
            df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
        else:
            FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(FEEDBACK_FILE, index=False)

        return "Feedback saved successfully."
    except Exception as e:
        return f"Error saving feedback: {e}"

def prepare_feedback_data():
    """
    Prepare feedback data for model training.

    Returns:
        Dataset: A Hugging Face Dataset object with processed feedback data.
    """
    if not FEEDBACK_FILE.exists():
        raise FileNotFoundError("Feedback data file does not exist.")
    
    feedback_data = pd.read_csv(FEEDBACK_FILE)
    
    # Assign labels based on feedback: "yes" -> accurate, "no" -> inaccurate
    feedback_data["label"] = feedback_data["feedback"].apply(lambda x: 1 if x == "yes" else 0)
    
    # Convert to Hugging Face Dataset format
    return Dataset.from_pandas(feedback_data[["text", "label"]])

def fine_tune_transformer_model():
    """
    Fine-tune a transformer-based model using feedback data.

    Returns:
        str: A message indicating the success or failure of the training process.
    """
    try:
        # Load feedback data
        feedback_dataset = prepare_feedback_data()

        # Split into train and test datasets
        train_test_split = feedback_dataset.train_test_split(test_size=0.2)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]

        # Define the model and tokenizer
        model_name = "distilbert-base-uncased"
        tokenizer = pipeline("sentiment-analysis", model=model_name).tokenizer
        
        # Tokenize datasets
        def tokenize_data(batch):
            return tokenizer(batch["text"], padding="max_length", truncation=True)

        train_dataset = train_dataset.map(tokenize_data, batched=True)
        eval_dataset = eval_dataset.map(tokenize_data, batched=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01
        )

        # Define the trainer
        trainer = Trainer(
            model=pipeline("sentiment-analysis", model=model_name).model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Train the model
        trainer.train()
        return "Transformer model fine-tuned successfully using feedback data."

    except Exception as e:
        return f"Error during training: {e}"

def learn_from_feedback():
    """
    Uses feedback data to improve sentiment learning pipeline.

    Returns:
        str: A message about feedback processing status.
    """
    if not FEEDBACK_FILE.exists():
        return "No feedback data to train on."

    try:
        # Fine-tune the transformer model using feedback data
        training_status = fine_tune_transformer_model()
        return training_status
    except Exception as e:
        return f"Error during learning from feedback: {e}"
