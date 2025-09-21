# app.py

from flask import Flask, render_template, request, jsonify
from legal_rag import get_document_summary
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['document']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_stream = io.BytesIO(file.read())
        try:
            summary, original_text = get_document_summary(file_stream)
            return jsonify({
                'summary': summary,
                'original_text': original_text
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # You need to set your Hugging Face API token in a .env file
    # HUGGINGFACEHUB_API_TOKEN="<your_token>"
    # You can also set it in your environment
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise ValueError("Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
    
    app.run(debug=True)