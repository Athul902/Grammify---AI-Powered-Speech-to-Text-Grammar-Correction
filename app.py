from flask import Flask, request, jsonify, render_template, send_file
import whisper
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
from PyPDF2 import PdfReader

app = Flask(__name__)

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load Grammar Correction Model
t5_model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction")
t5_tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/speech")
def ppeech_page():
    return render_template("speech.html")

@app.route("/text")
def text_page():
    return render_template("text.html")

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    audio_file = request.files["audio"]
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    # Transcribe audio
    result = whisper_model.transcribe(audio_path)
    transcribed_text = result["text"]

    # Grammar Correction
    input_text = "grammar: " + transcribed_text
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = t5_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    corrected_text = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    os.remove(audio_path)

    return jsonify({
        "original_text": transcribed_text,
        "transcription": corrected_text
    })

@app.route("/correct_text", methods=["POST"])
def correct_text():
    corrected_text = ""

    if "text_input" in request.form:
        text = request.form.get("text_input", "")
        corrected_text = correct_grammar(text)
    
    elif "file" in request.files:
        file = request.files["file"]
        if file.filename.endswith(".pdf"):
            reader = PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            text = file.read().decode("utf-8")
        
        corrected_text = correct_grammar(text)

    # Save corrected text for download
    output_filename = "corrected_text.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(corrected_text)

    return jsonify({"corrected_text": corrected_text, "download_url": "/download"})

@app.route("/download")
def download_file():
    return send_file("corrected_text.txt", as_attachment=True)

def correct_grammar(text):
    input_text = "grammar: " + text
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = t5_model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    app.run(debug=True)
