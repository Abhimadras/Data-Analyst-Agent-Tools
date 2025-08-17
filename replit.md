# Data Analyst Agent API

## Overview

A Flask-based REST API that acts as an intelligent Data Analyst Agent. The system processes natural language questions from text files and performs AI-powered data analysis on uploaded datasets. It combines OpenAI's GPT-4o-mini with Python data science libraries to provide automated analysis, visualization generation, and structured responses. The agent supports multiple data formats (CSV, JSON, Parquet) and can perform web scraping for external data sources.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Web Framework Architecture
- **Framework**: Flask with RESTful API design
- **Endpoints**: 
  - Health check at `/` 
  - Main analysis at `/api/` (POST only)
- **Request Handling**: Multipart form-data for file uploads with automatic file type detection
- **Response Format**: Always returns JSON (objects or arrays) with consistent error handling

### AI Integration
- **Analysis Engine**: Cost-free keyword-based question interpretation (no external APIs required)
- **Processing Flow**: Questions → Keyword analysis → Data analysis → Structured response
- **Timeout Protection**: 170-second execution limit with graceful failure handling

### Data Processing Pipeline
- **File Support**: CSV, JSON, Parquet for datasets; PNG, JPG for images; TXT for questions
- **Data Engine**: DuckDB for SQL operations, Pandas for data manipulation
- **Analysis Libraries**: NumPy, Scikit-learn for statistical analysis and machine learning
- **Web Scraping**: Requests + Trafilatura for external data sourcing

### Visualization System
- **Library**: Matplotlib with non-interactive backend (Agg)
- **Output Format**: Base64-encoded images under 100KB
- **Chart Types**: Supports various matplotlib visualizations
- **Memory Management**: In-memory image processing with automatic cleanup

### Error Handling & Safety
- **Timeout Management**: Signal-based timeout with custom exception handling
- **File Security**: Werkzeug secure filename validation
- **Logging**: Comprehensive logging throughout the application
- **Graceful Degradation**: Returns structured error responses on failure

### Deployment Architecture
- **Containerization**: Docker-ready with production configuration
- **Environment Management**: Environment variables for API keys and secrets
- **Testing**: Pytest-based test suite with fixtures and mocking

## External Dependencies

### Analysis Engine
- **Cost-Free Intelligence**: Built-in keyword-based question interpretation system
- **No External Dependencies**: Eliminates API costs and network dependencies

### Data Processing Libraries
- **pandas**: DataFrame operations and data manipulation
- **numpy**: Numerical computations and array operations
- **duckdb**: In-memory SQL database for complex queries
- **scikit-learn**: Machine learning algorithms and statistical analysis

### Web & Network
- **requests**: HTTP client for web scraping and API calls
- **trafilatura**: Web content extraction and text processing
- **Flask**: Web framework and HTTP request handling
- **Werkzeug**: WSGI utilities and security helpers

### Visualization & File Handling
- **matplotlib**: Chart generation and data visualization
- **tempfile**: Secure temporary file management

### Testing Framework
- **pytest**: Unit testing framework with fixtures and mocking capabilities