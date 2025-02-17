import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the English-to-Urdu translation model from Hugging Face
model_name = "Helsinki-NLP/opus-mt-en-ur"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_english_to_urdu(text):
    """Translate input English text to Urdu."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return translated_text

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("<h1 align='center'>🌍 English to Urdu Translator</h1>")
    
    with gr.Row():
        input_text = gr.Textbox(label="Enter English Text", placeholder="Type here...")
        output_text = gr.Textbox(label="Translated Urdu Text", interactive=False)

    translate_button = gr.Button("Translate")

    translate_button.click(translate_english_to_urdu, inputs=input_text, outputs=output_text)

# Launch the app
demo.launch()
