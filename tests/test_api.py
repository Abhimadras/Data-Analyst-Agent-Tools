import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from app import app
import pandas as pd

@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_questions_file():
    """Create a temporary questions file"""
    content = "What is the average value?\nShow me a correlation analysis."
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file"""
    df = pd.DataFrame({
        'value1': [1, 2, 3, 4, 5],
        'value2': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'B', 'A']
    })
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data == {"status": "ok"}

def test_api_no_files(client):
    """Test API endpoint with no files"""
    response = client.post('/api/')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "No files uploaded" in data["error"]

def test_api_no_txt_file(client, sample_csv_file):
    """Test API endpoint with no .txt file"""
    with open(sample_csv_file, 'rb') as f:
        response = client.post('/api/', data={
            'data.csv': f
        })
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "No .txt file found" in data["error"]

def test_api_empty_questions_file(client):
    """Test API endpoint with empty questions file"""
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        f.write(b'')
        f.flush()
        f.seek(0)
        
        response = client.post('/api/', data={
            'questions.txt': f
        })
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data
    assert "empty" in data["error"].lower()

@patch('data_analyzer.DataAnalyzer.analyze')
def test_api_with_questions_file_only(mock_analyze, client, sample_questions_file):
    """Test API endpoint with questions file only"""
    mock_analyze.return_value = {
        "question": "What is the average value?",
        "answer": "Test answer",
        "insights": ["Test insight"]
    }
    
    with open(sample_questions_file, 'rb') as f:
        response = client.post('/api/', data={
            'myquestions.txt': f  # Test with different filename
        })
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "question" in data
    assert "answer" in data
    mock_analyze.assert_called_once()

@patch('data_analyzer.DataAnalyzer.analyze')
def test_api_with_questions_and_csv(mock_analyze, client, sample_questions_file, sample_csv_file):
    """Test API endpoint with questions file and CSV data"""
    mock_analyze.return_value = {
        "question": "What is the correlation?",
        "answer": "Strong positive correlation",
        "visualization": "data:image/png;base64,test123"
    }
    
    with open(sample_questions_file, 'rb') as qf, open(sample_csv_file, 'rb') as cf:
        response = client.post('/api/', data={
            'questions.txt': qf,
            'data.csv': cf
        })
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "answer" in data
    mock_analyze.assert_called_once()
    
    # Verify that CSV data was processed
    call_args = mock_analyze.call_args
    files_dict = call_args[0][1]  # Second argument (files)
    assert any('csv' in filename.lower() for filename in files_dict.keys())

@patch('data_analyzer.DataAnalyzer.analyze')
def test_api_multiple_questions(mock_analyze, client):
    """Test API endpoint with multiple questions"""
    mock_analyze.return_value = [
        {"question": "Question 1", "answer": "Answer 1"},
        {"question": "Question 2", "answer": "Answer 2"}
    ]
    
    questions_content = "Question 1\nQuestion 2"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
        f.write(questions_content)
        f.flush()
        f.seek(0)
        
        with open(f.name, 'rb') as qf:
            response = client.post('/api/', data={
                'questions.txt': qf
            })
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) == 2

@patch('data_analyzer.DataAnalyzer.analyze')
def test_api_analysis_error(mock_analyze, client, sample_questions_file):
    """Test API endpoint when analysis fails"""
    mock_analyze.return_value = {"error": "Analysis failed: Test error"}
    
    with open(sample_questions_file, 'rb') as f:
        response = client.post('/api/', data={
            'questions.txt': f
        })
    
    assert response.status_code == 200  # API should return 200 but with error in JSON
    data = json.loads(response.data)
    assert "error" in data
    assert "Analysis failed" in data["error"]

def test_api_json_file_processing(client, sample_questions_file):
    """Test API endpoint with JSON file processing"""
    json_data = {"records": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as jf:
        json.dump(json_data, jf)
        jf.flush()
        
        try:
            with patch('data_analyzer.DataAnalyzer.analyze') as mock_analyze:
                mock_analyze.return_value = {"answer": "Test"}
                
                with open(sample_questions_file, 'rb') as qf, open(jf.name, 'rb') as jsonf:
                    response = client.post('/api/', data={
                        'questions.txt': qf,
                        'data.json': jsonf
                    })
                
                assert response.status_code == 200
                # Verify JSON was processed
                call_args = mock_analyze.call_args
                files_dict = call_args[0][1]
                assert any('json' in filename.lower() for filename in files_dict.keys())
        finally:
            os.unlink(jf.name)

def test_404_endpoint(client):
    """Test 404 error handling"""
    response = client.get('/nonexistent')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert "error" in data
    assert "not found" in data["error"].lower()

@patch('data_analyzer.DataAnalyzer.__init__')
def test_api_initialization_error(mock_init, client, sample_questions_file):
    """Test API when DataAnalyzer initialization fails"""
    mock_init.side_effect = Exception("Initialization failed")
    
    with open(sample_questions_file, 'rb') as f:
        response = client.post('/api/', data={
            'questions.txt': f
        })
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert "error" in data

def test_file_extension_detection(client):
    """Test that .txt files are detected regardless of name"""
    questions_content = "What is the test result?"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt') as f:
        f.write(questions_content)
        f.flush()
        
        with patch('data_analyzer.DataAnalyzer.analyze') as mock_analyze:
            mock_analyze.return_value = {"answer": "Test"}
            
            # Test with unusual filename
            with open(f.name, 'rb') as qf:
                response = client.post('/api/', data={
                    'my_unusual_filename_123.txt': qf
                })
            
            assert response.status_code == 200
            mock_analyze.assert_called_once()
            
            # Verify questions content was passed correctly
            call_args = mock_analyze.call_args
            questions_arg = call_args[0][0]  # First argument (questions)
            assert "test result" in questions_arg.lower()

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
