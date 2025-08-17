# Data Analyst Agent API

A Flask-based API that acts as a Data Analyst Agent, capable of processing natural language questions and performing intelligent data analysis with visualization generation - **completely cost-free with no API dependencies**.

## Features

- 🤖 **Intelligent Analysis**: Built-in keyword-based question interpretation (no OpenAI API required)
- 📊 **Data Processing**: Supports CSV, JSON, and Parquet file formats with automatic preprocessing
- 📈 **Visualization**: Generates matplotlib charts encoded as base64 images (under 100KB)
- 🌐 **Web Scraping**: Capable of fetching and analyzing data from web sources
- 🛡️ **Robust Error Handling**: Comprehensive error handling with 3-minute timeout protection
- 💰 **Cost-Free**: No external API keys or paid services required
- 🚀 **Production Ready**: Containerized with Docker and ready for deployment

## API Endpoints

### Health Check
```bash
GET /
# Response: {"status": "ok"}
```

### Data Analysis
```bash
POST /api/
# Accepts multipart form-data with:
# - One .txt file containing questions (any filename ending in .txt)
# - Zero or more data files (CSV, JSON, Parquet, PNG, JPG)
```

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd data-analyst-agent

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### 2. Basic Usage

Create a questions file:
```bash
echo "What is the average value in the data?
Show me a correlation analysis.
Create a visualization of the distribution." > questions.txt
```

Create sample data:
```bash
echo "value1,value2,category
1,10,A
2,20,B
3,30,A
4,40,B
5,50,A" > data.csv
```

Make a request:
```bash
curl -X POST http://localhost:5000/api/ \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv"
```

## Supported Question Types

The system automatically detects question intent and provides appropriate analysis:

### Statistical Analysis
- **Keywords**: average, mean, sum, count, min, max
- **Example**: "What is the average value in the dataset?"

### Correlation Analysis
- **Keywords**: correlation, relationship, relate
- **Example**: "Show me the correlation between variables"

### Regression Analysis
- **Keywords**: regression, predict, forecast, trend
- **Example**: "Can you predict the trend in this data?"

### Data Visualization
- **Keywords**: plot, chart, graph, visualize, show
- **Example**: "Create a plot showing the distribution"

### Comparative Analysis
- **Keywords**: compare, comparison, versus, difference
- **Example**: "Compare values across different categories"

## Sample Response

```json
{
  "question": "What is the correlation between value1 and value2?",
  "answer": "The strongest correlation is between value1 vs value2 with a coefficient of 1.0 (very strong)",
  "insights": [
    "Found 1 significant correlations in the data",
    "Data has 0 missing values across 0 columns"
  ],
  "calculations": {
    "statistics": {
      "value1": {"mean": 3.0, "std": 1.581, "min": 1.0, "max": 5.0, "count": 5},
      "value2": {"mean": 30.0, "std": 15.811, "min": 10.0, "max": 50.0, "count": 5}
    },
    "correlations": [
      {
        "variables": "value1 vs value2",
        "correlation": 1.0,
        "strength": "very strong"
      }
    ]
  },
  "visualization": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

## Supported File Formats

### Data Files
- **CSV**: Comma-separated values with automatic type detection
- **JSON**: Structured data automatically flattened to DataFrame
- **Parquet**: Columnar data format with efficient compression

### Image Files
- **PNG, JPG**: For image analysis (future feature)

### Questions File
- **TXT**: Plain text file with one question per line

## Docker Deployment

```bash
# Build the image
docker build -t data-analyst-agent .

# Run the container
docker run -p 5000:5000 data-analyst-agent
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## Architecture

The system uses intelligent keyword-based analysis to interpret questions without requiring external AI services:

1. **Question Parsing**: Analyzes keywords to determine analysis type
2. **Data Processing**: Automatically loads and preprocesses uploaded files
3. **Analysis Engine**: Performs statistical analysis, correlations, and regression
4. **Visualization**: Generates matplotlib charts as base64 images
5. **Response**: Returns structured JSON with insights and supporting data

## Error Handling

The API includes comprehensive error handling:
- **File validation**: Ensures required .txt file is present
- **Data validation**: Handles malformed CSV/JSON files gracefully
- **Timeout protection**: 3-minute maximum execution time
- **Memory management**: Automatic cleanup of temporary files
- **Structured errors**: Always returns JSON error responses

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues or questions:
1. Check the test files for usage examples
2. Review error messages in API responses
3. Create an issue in the repository

---

**Note**: This is a cost-free alternative to AI-powered data analysis services. While it doesn't use large language models, it provides intelligent analysis through keyword matching and statistical computing.