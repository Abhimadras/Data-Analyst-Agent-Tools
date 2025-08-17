# Running Data Analyst Agent Locally

This guide will help you run the cost-free Data Analyst Agent on your local PC.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Quick Setup

### 1. Download the Project Files

You'll need these files from the project:
- `app.py` - Flask application
- `data_analyzer.py` - Core analysis engine
- `main.py` - Entry point
- `requirements.txt` - Dependencies (create this file - see below)

### 2. Create Requirements File

Create a `requirements.txt` file with these dependencies:

```
Flask==2.3.3
pandas==2.1.1
numpy==1.24.3
matplotlib==3.7.2
requests==2.31.0
trafilatura==1.6.2
duckdb==0.8.1
scikit-learn==1.3.0
beautifulsoup4==4.12.2
Werkzeug==2.3.7
```

### 3. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```

The server will start on `http://localhost:5000`

## Testing the API

### Test with curl (if available):

```bash
# Create a test question file
echo "What is the total sales? Which region has highest sales?" > questions.txt

# Create sample CSV data
echo "date,region,sales
2024-01-01,East,100
2024-01-02,West,200
2024-01-03,East,150" > sample.csv

# Test the API
curl -X POST http://localhost:5000/api/ \
  -F "questions.txt=@questions.txt" \
  -F "sample.csv=@sample.csv"
```

### Test with Python:

```python
import requests

# Test files
files = {
    'questions.txt': ('questions.txt', 'What is the total sales?'),
    'sample.csv': ('sample.csv', 'date,region,sales\n2024-01-01,East,100\n2024-01-02,West,200')
}

response = requests.post('http://localhost:5000/api/', files=files)
print(response.json())
```

### Test with a web browser:

1. Open `http://localhost:5000` in your browser
2. You should see a health check message
3. Use the POST endpoint `/api/` with file uploads

## Features

✅ **Cost-Free**: No API keys required - uses built-in intelligence
✅ **File Support**: CSV, JSON, Parquet, TXT files
✅ **Web Scraping**: Automatic Wikipedia data extraction
✅ **Visualizations**: Generates charts as base64 images under 100KB
✅ **Multiple Questions**: Handles multiple analysis requests
✅ **Structured JSON**: Supports evaluation-format responses

## API Endpoints

### Health Check
- **GET** `/` - Returns system status

### Data Analysis
- **POST** `/api/` - Main analysis endpoint
  - Accepts multipart form data with files
  - Returns JSON with analysis results
  - Supports both individual questions and structured JSON requests

## File Formats Supported

- **Questions**: `.txt` files with natural language questions
- **Data**: `.csv`, `.json`, `.parquet` files
- **Images**: `.png`, `.jpg` for image analysis (if implemented)

## Example Usage

The system excels at:
- Sales data analysis
- Statistical calculations
- Correlation analysis
- Data visualization generation
- Web scraping for external data
- Structured JSON responses for automated evaluation

## Troubleshooting

### Port Already in Use
If port 5000 is busy, modify `main.py`:
```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)  # Change port to 8080
```

### Missing Dependencies
Make sure all packages in `requirements.txt` are installed:
```bash
pip install -r requirements.txt
```

### File Upload Issues
Ensure files are properly formatted and contain valid data.

## Production Deployment

For production use:
1. Set `debug=False` in `main.py`
2. Use a production WSGI server like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn --bind 0.0.0.0:5000 main:app
   ```
3. Configure proper logging and error handling
4. Set up reverse proxy (nginx) if needed

## Cost Benefits

This system provides enterprise-level data analysis capabilities without:
- OpenAI API costs
- External service dependencies
- Complex infrastructure requirements
- Ongoing subscription fees

Perfect for local development, testing, and cost-sensitive production environments.