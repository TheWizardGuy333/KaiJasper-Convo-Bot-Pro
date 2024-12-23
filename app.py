import streamlit as st
import nltk
from utils.sentiment_analysis import analyze_sentiment, learn_from_feedback
from utils.image_processing import process_image
from utils.summarization import summarize_text
from utils.visualizations import plot_sentiment_distribution
from utils.data_retention import save_chat, get_chat_history, save_feedback
from utils.nlg import generate_response  # For AI replies using Hugging Face Transformers
from config import API_KEY

# Ensure necessary NLTK data is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

st.set_page_config(page_title="KaiJasper Convo Bot Pro", layout="wide")

st.title("KaiJasper Convo Bot Pro")
st.markdown("An AI-driven assistant for text and image processing, sentiment analysis, and analytics.")

# Chatbot Section
st.header("Chatbot")
conversation_history = st.session_state.get("conversation_history", [])

user_input = st.text_area("Enter your message:")

# Analyze Sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = analyze_sentiment(user_input)
        st.write(f"Polarity: {sentiment['polarity']:.2f}, Subjectivity: {sentiment['subjectivity']:.2f}")
        save_chat(user_input, sentiment)

        feedback = st.radio("Was this response accurate?", ["Yes", "No"])
        if st.button("Submit Feedback"):
            save_feedback(user_input, sentiment, feedback)
            st.success("Feedback saved successfully!")
    else:
        st.warning("Please enter valid text for sentiment analysis.")

# AI Reply Generation
if st.button("Get AI Reply"):
    if user_input.strip():
        # Generate a reply using Hugging Face Transformers
        ai_reply = generate_response(user_input)
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "bot", "content": ai_reply})
        st.session_state.conversation_history = conversation_history

        # Display the conversation
        for turn in conversation_history:
            role = "You" if turn["role"] == "user" else "KaiJasper Bot"
            st.write(f"**{role}:** {turn['content']}")
    else:
        st.warning("Please enter text for the AI to respond.")

# Text Summarization
if st.button("Summarize Text"):
    if user_input.strip():
        summary_method = st.selectbox("Choose Summarization Method", ["simple", "ml"])
        num_sentences = st.slider("Number of Sentences", min_value=1, max_value=5, value=2)
        summary = summarize_text(user_input, method=summary_method, num_sentences=num_sentences)
        st.write("Summary:", summary)
    else:
        st.warning("Please enter text to summarize.")

# Image Processing Section
st.header("Image Processing")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

if uploaded_file and st.button("Process Image"):
    processed_image_url = process_image(uploaded_file, API_KEY)
    if processed_image_url:
        st.image(processed_image_url, caption="Processed Image", use_container_width=True)
    else:
        st.error("Image processing failed. Please try again.")

# Visualization Section
st.header("Analytics Dashboard")
if st.button("Show Sentiment Distribution"):
    try:
        plot_sentiment_distribution()
    except FileNotFoundError:
        st.error("No sentiment data available to display.")

# Chat History Section
st.header("Chat History")
if st.button("Load Chat History"):
    chat_history = get_chat_history()
    if isinstance(chat_history, str):
        st.warning(chat_history)
    else:
        st.write(chat_history)

# Feedback Learning Section
st.header("Learning from Feedback")
if st.button("Train on Feedback Data"):
    feedback_result = learn_from_feedback()
    st.success(feedback_result)