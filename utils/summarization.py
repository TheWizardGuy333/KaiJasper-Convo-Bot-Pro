import nltk
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def summarize_text(text, method="simple", num_sentences=2):
    """
    Summarize the given text using the specified method.

    Args:
        text (str): The input text to summarize.
        method (str): Summarization method - "simple" or "lex_rank".
        num_sentences (int): Number of sentences to include in the summary.

    Returns:
        str: The summarized text.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")

    if method == "simple":
        # Simple summarization using TextBlob
        blob = TextBlob(text)
        sentences = blob.sentences[:num_sentences]
        return " ".join(str(sentence) for sentence in sentences)
    
    elif method == "lex_rank":
        # Advanced summarization using Sumy
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)

    else:
        raise ValueError(f"Unknown summarization method: {method}")