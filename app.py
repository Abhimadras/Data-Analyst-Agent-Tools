import os
import tempfile
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from data_analyzer import DataAnalyzer
import pandas as pd
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Initialize the data analyzer
analyzer = DataAnalyzer()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/api/', methods=['POST'])
def analyze_data():
    """Main analysis endpoint"""
    try:
        # Check if any files were uploaded
        if not request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        # Find the .txt file (questions file)
        questions_file = None
        questions_filename = None
        
        for filename, file in request.files.items():
            if filename.lower().endswith('.txt'):
                questions_file = file
                questions_filename = filename
                break
        
        if not questions_file:
            return jsonify({"error": "No .txt file found in upload"}), 400
        
        # Create temporary directory for file processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save questions file
            questions_path = os.path.join(temp_dir, 'questions.txt')
            questions_file.save(questions_path)
            
            # Read questions content
            with open(questions_path, 'r', encoding='utf-8') as f:
                questions_content = f.read().strip()
            
            if not questions_content:
                return jsonify({"error": "Questions file is empty"}), 400
            
            # Process other uploaded files
            uploaded_files = {}
            
            for filename, file in request.files.items():
                if filename == questions_filename:
                    continue  # Skip the questions file
                
                if not file.filename or file.filename == '':
                    continue
                
                # Save file temporarily
                secure_name = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, secure_name)
                file.save(file_path)
                
                # Process different file types
                try:
                    if secure_name.lower().endswith('.csv'):
                        df = pd.read_csv(file_path)
                        # Convert numeric columns
                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        uploaded_files[secure_name] = df
                        
                    elif secure_name.lower().endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                        else:
                            df = pd.json_normalize(data)
                        # Convert numeric columns
                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        uploaded_files[secure_name] = df
                        
                    elif secure_name.lower().endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                        uploaded_files[secure_name] = df
                        
                    else:
                        # For other file types (images, etc.), store the file path
                        uploaded_files[secure_name] = file_path
                        
                except Exception as e:
                    logger.warning(f"Failed to process file {secure_name}: {str(e)}")
                    # Continue processing other files
                    continue
            
            # Perform analysis
            try:
                result = analyzer.analyze(questions_content, uploaded_files)
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Analysis failed: {str(e)}")
                return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        return jsonify({"error": f"Request processing failed: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
