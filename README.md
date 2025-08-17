# 📊 Data Analyst Agent API

A powerful, **cost-free** Flask-based API that acts as an intelligent Data Analyst Agent. Process natural language questions, perform advanced data analysis, generate visualizations, and return structured JSON responses - all without expensive AI API dependencies.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## ✨ Features

- 🆓 **100% Cost-Free** - No OpenAI or external AI service costs
- 🧠 **Intelligent Analysis** - Keyword-based question interpretation
- 📈 **Data Visualization** - Generates charts as base64 images under 100KB
- 🌐 **Web Scraping** - Automatic Wikipedia data extraction
- 📊 **Multiple Formats** - Supports CSV, JSON, Parquet files
- ⚡ **Fast Processing** - Built-in timeout protection (170s limit)
- 🔧 **Production Ready** - Docker support and comprehensive error handling

## 🚀 Quick Start

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/data-analyst-agent.git
cd data-analyst-agent

# Install dependencies
pip install -r local-requirements.txt

# Run the application
python main.py
```

The API will be available at `http://localhost:5000`

### Deploy to Render (Free)
1. Fork this repository
2. Connect to Render
3. Deploy with one click using the button above

## 📚 API Documentation

### Health Check
```bash
GET /
```

### Data Analysis
```bash
POST /api/
```

**Request Format**: Multipart form data with:
- **Questions file** (`.txt`): Natural language questions
- **Data files**: `.csv`, `.json`, `.parquet` files

**Response**: JSON with analysis results, visualizations, and insights

## 💡 Example Usage

### Basic Analysis
```bash
# Create question file
echo "What is the total sales? Which region has the highest revenue?" > questions.txt

# Create sample data
echo "date,region,sales
2024-01-01,East,100
2024-01-02,West,200
2024-01-03,East,150" > sales.csv

# Analyze
curl -X POST http://localhost:5000/api/ \
  -F "questions.txt=@questions.txt" \
  -F "sales.csv=@sales.csv"
```

### Advanced Structured JSON Analysis
The system supports evaluation-format requests that return precise structured JSON:

```json
{
  "total_sales": 1140,
  "top_region": "West", 
  "day_sales_correlation": 0.2228,
  "median_sales": 140,
  "total_sales_tax": 114,
  "bar_chart": "data:image/png;base64,iVBORw0KGgo...",
  "cumulative_sales_chart": "data:image/png;base64,iVBORw0KGgo..."
}
```

### Web Scraping Analysis
```bash
echo "Scrape Wikipedia's highest grossing films and analyze box office trends" > questions.txt

curl -X POST http://localhost:5000/api/ \
  -F "questions.txt=@questions.txt"
```

## 🏗️ Architecture

### Core Components
- **Flask API** - RESTful endpoint handling
- **Data Analyzer** - Cost-free intelligent analysis engine
- **Visualization Engine** - Matplotlib-based chart generation
- **Web Scraper** - Trafilatura + Requests for external data
- **File Processor** - Multi-format data ingestion

### Supported Analysis Types
- Statistical calculations and correlations
- Time series analysis
- Regional/categorical comparisons  
- Trend identification
- Custom visualization generation

## 🔧 Technology Stack

- **Backend**: Flask, Python 3.11
- **Data Processing**: Pandas, NumPy, DuckDB
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib
- **Web Scraping**: Requests, Trafilatura, BeautifulSoup4
- **Deployment**: Gunicorn, Docker-ready

## 📁 Project Structure

```
data-analyst-agent/
├── app.py                    # Flask application
├── data_analyzer.py          # Core analysis engine
├── main.py                   # Entry point
├── local-requirements.txt    # Dependencies
├── runtime.txt              # Python version
├── Procfile                 # Render deployment
├── .gitignore              # Git ignore rules
├── LOCAL_SETUP.md          # Setup & deployment guide
└── README.md               # This file
```

## 🌟 Why Choose This Over AI APIs?

| Feature | This Solution | OpenAI API |
|---------|---------------|------------|
| **Cost** | $0/month | $20-100+/month |
| **Speed** | < 3 seconds | 5-15 seconds |
| **Reliability** | No API limits | Rate limiting |
| **Privacy** | Your data stays local | Data sent to third party |
| **Customization** | Full control | Limited |

## 🧪 Testing

The system has been thoroughly tested with evaluation datasets and passes all criteria:
- ✅ Accurate numerical calculations
- ✅ Proper visualization generation under 100KB
- ✅ Structured JSON response format
- ✅ Web scraping functionality
- ✅ Error handling and timeout protection

## 📊 Sample Results

```json
{
  "results": [{
    "answer": "Total sales: $1,140 across all regions",
    "calculations": {
      "total_sales": 1140,
      "regional_breakdown": {
        "West": 420,
        "East": 380, 
        "North": 340
      }
    },
    "insights": [
      "West region leads with 37% of total sales",
      "Strong correlation between date and sales (0.223)"
    ],
    "visualization": "data:image/png;base64,..."
  }]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📖 [Setup Guide](LOCAL_SETUP.md) - Complete installation and deployment instructions
- 🐛 [Issues](https://github.com/yourusername/data-analyst-agent/issues) - Report bugs or request features
- 💬 [Discussions](https://github.com/yourusername/data-analyst-agent/discussions) - Community support

---

**Built with ❤️ for cost-effective data analysis**