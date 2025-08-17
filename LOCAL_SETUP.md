# Data Analyst Agent - Local Setup & Deployment Guide

This guide covers running the cost-free Data Analyst Agent locally and deploying it publicly via GitHub and Render.

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

---

# GitHub & Render Deployment

## Part 1: Prepare for GitHub

### 1. Create Deployment Files

Create these additional files in your project root:

**`runtime.txt`** (specify Python version):
```
python-3.11.0
```

**`Procfile`** (for Render deployment):
```
web: gunicorn --bind 0.0.0.0:$PORT main:app
```

**`.gitignore`**:
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
.env
instance/
.pytest_cache/
.coverage
htmlcov/
.DS_Store
*.log
```

**`app.json`** (for easy deployment):
```json
{
  "name": "Data Analyst Agent API",
  "description": "Cost-free intelligent data analysis API with visualization generation",
  "repository": "https://github.com/yourusername/data-analyst-agent",
  "keywords": ["flask", "data-analysis", "api", "visualization", "cost-free"],
  "env": {
    "FLASK_ENV": {
      "description": "Flask environment",
      "value": "production"
    }
  },
  "formation": {
    "web": {
      "quantity": 1,
      "size": "free"
    }
  }
}
```

### 2. Update main.py for Production

Modify `main.py` to handle port configuration:

```python
import os
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
```

### 3. Add gunicorn to dependencies

Update your `local-requirements.txt` to include:
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
gunicorn==21.2.0
```

## Part 2: Deploy to GitHub

### 1. Initialize Git Repository
```bash
git init
git add .
git commit -m "Initial commit: Data Analyst Agent API"
```

### 2. Create GitHub Repository
1. Go to GitHub.com
2. Click "New Repository"
3. Name it `data-analyst-agent` (or your preferred name)
4. Don't initialize with README (you already have files)
5. Click "Create Repository"

### 3. Push to GitHub
```bash
git remote add origin https://github.com/yourusername/data-analyst-agent.git
git branch -M main
git push -u origin main
```

## Part 3: Deploy to Render

### 1. Connect GitHub to Render
1. Go to [render.com](https://render.com)
2. Sign up/login
3. Click "New +" → "Web Service"
4. Connect your GitHub account
5. Select your `data-analyst-agent` repository

### 2. Configure Render Deployment
**Build Settings:**
- **Environment**: Python 3
- **Build Command**: `pip install -r local-requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT main:app`

**Environment Variables:**
- `FLASK_ENV`: `production`
- `PYTHON_VERSION`: `3.11.0`

### 3. Deploy
1. Click "Create Web Service"
2. Render will automatically build and deploy
3. You'll get a public URL like: `https://your-app-name.onrender.com`

## Part 4: Test Public Deployment

### Test the deployed API:
```bash
# Test health check
curl https://your-app-name.onrender.com/

# Test analysis endpoint
curl -X POST https://your-app-name.onrender.com/api/ \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv"
```

## Part 5: GitHub Repository Structure

Your final repository should look like:
```
data-analyst-agent/
├── app.py                    # Flask application
├── data_analyzer.py          # Core analysis engine  
├── main.py                   # Entry point
├── local-requirements.txt    # Dependencies
├── runtime.txt              # Python version
├── Procfile                 # Render deployment config
├── app.json                 # Deployment metadata
├── .gitignore              # Git ignore rules
├── LOCAL_SETUP.md          # This documentation
├── README.md               # Project description
└── LICENSE                 # License file
```

## Part 6: Maintenance & Updates

### Update Deployment:
```bash
# Make changes to your code
git add .
git commit -m "Update: description of changes"
git push origin main
```

Render will automatically redeploy when you push to GitHub.

### Monitor Deployment:
- Check Render dashboard for logs and metrics
- Monitor API performance and usage
- Set up alerts for downtime

## Benefits of This Deployment

✅ **Free Hosting**: Render free tier supports this lightweight API  
✅ **Automatic Deployments**: Updates via GitHub push  
✅ **Public Access**: Anyone can use your API  
✅ **Scalable**: Can upgrade Render plan if needed  
✅ **Cost-Free Operation**: No ongoing API costs  
✅ **Professional URL**: Custom domain support available  

## API Documentation for Users

Once deployed, users can access:
- **Health Check**: `GET https://your-app.onrender.com/`
- **Data Analysis**: `POST https://your-app.onrender.com/api/`

The API accepts multipart form data with text files (questions) and data files (CSV/JSON/Parquet) and returns intelligent analysis with visualizations.

Perfect for local development, testing, and cost-sensitive production environments.