from transformers import pipeline

# Initialize Hugging Face model pipeline
response_generator = pipeline("text-generation", model="gpt2")

def generate_response(prompt, max_length=150):
    """
    Generate a response using a Hugging Face Transformers model.

    Args:
        prompt (str): User input or conversation context.
        max_length (int): Maximum length of the generated response.

    Returns:
        str: Generated response.
    """
    try:
        result = response_generator(prompt, max_length=max_length, num_return_sequences=1)
        return result[0]["generated_text"]
    except Exception as e:
        return f"Error generating response: {e}"