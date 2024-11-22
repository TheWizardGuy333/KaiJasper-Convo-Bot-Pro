from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def summarize_text(text, method="simple", num_sentences=2):
    if method == "simple":
        blob = TextBlob(text)
        sentences = blob.sentences[:num_sentences]
        return " ".join(str(sentence) for sentence in sentences)

    elif method == "ml":
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)

    else:
        raise ValueError("Invalid summarization method. Choose 'simple' or 'ml'.")
