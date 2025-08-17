import os
import json
import logging
import base64
import io
import re
import signal
from contextlib import contextmanager
import requests
import trafilatura
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import duckdb

logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout handling"""
    def signal_handler(signum, frame):
        raise TimeoutException("Operation timed out")
    
    # Set the signal handler and a timeout alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm

class DataAnalyzer:
    def __init__(self):
        self.analysis_keywords = {
            'average': ['average', 'mean', 'avg'],
            'correlation': ['correlation', 'correlate', 'relationship', 'relate'],
            'regression': ['regression', 'predict', 'forecast', 'trend'],
            'distribution': ['distribution', 'histogram', 'spread', 'range'],
            'comparison': ['compare', 'comparison', 'versus', 'vs', 'difference'],
            'summary': ['summary', 'summarize', 'overview', 'describe'],
            'visualization': ['plot', 'chart', 'graph', 'visualize', 'show']
        }
        
    def analyze(self, questions: str, files: dict) -> dict:
        """Main analysis method"""
        try:
            with timeout(170):  # 170 seconds to leave buffer for response processing
                return self._perform_analysis(questions, files)
        except TimeoutException:
            return {"error": "Analysis timed out after 3 minutes"}
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _perform_analysis(self, questions: str, files: dict) -> dict:
        """Internal analysis method"""
        # Parse questions into individual questions
        question_lines = [q.strip() for q in questions.split('\n') if q.strip()]
        
        if not question_lines:
            return {"error": "No questions found"}
        
        # Analyze the intent of the questions using GPT-4o-mini
        analysis_plan = self._get_analysis_plan(question_lines, list(files.keys()))
        
        results = []
        
        for i, question in enumerate(question_lines):
            try:
                result = self._process_single_question(question, files, analysis_plan, i)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question {i}: {str(e)}")
                results.append({"question": question, "error": str(e)})
        
        # Return as array if multiple questions, or single object if one question
        if len(results) == 1:
            return results[0]
        else:
            return {"results": results}
    
    def _get_analysis_plan(self, questions: list, file_names: list) -> dict:
        """Analyze questions using keyword matching to determine the analysis approach"""
        plans = []
        
        for i, question in enumerate(questions):
            question_lower = question.lower()
            
            # Determine task type and analysis method based on keywords
            task_type = "data_analysis"
            analysis_method = "descriptive"
            visualization_needed = False
            visualization_type = "scatter"
            
            # Check for specific analysis types
            if any(keyword in question_lower for keyword in self.analysis_keywords['correlation']):
                analysis_method = "correlation"
                visualization_needed = True
                visualization_type = "scatter"
            elif any(keyword in question_lower for keyword in self.analysis_keywords['regression']):
                analysis_method = "regression"
                visualization_needed = True
                visualization_type = "scatter"
            elif any(keyword in question_lower for keyword in self.analysis_keywords['distribution']):
                analysis_method = "descriptive"
                visualization_needed = True
                visualization_type = "histogram"
            elif any(keyword in question_lower for keyword in self.analysis_keywords['comparison']):
                analysis_method = "comparison"
                visualization_needed = True
                visualization_type = "bar"
            elif any(keyword in question_lower for keyword in self.analysis_keywords['visualization']):
                visualization_needed = True
                
            # Check if visualization is explicitly requested
            if any(keyword in question_lower for keyword in self.analysis_keywords['visualization']):
                visualization_needed = True
                
            # Determine data source
            data_source = "uploaded_files" if file_names else "web_scraping"
            if "web" in question_lower or "online" in question_lower or "internet" in question_lower:
                data_source = "web_scraping"
                
            plans.append({
                "index": i,
                "task_type": task_type,
                "data_source": data_source,
                "analysis_method": analysis_method,
                "visualization_needed": visualization_needed,
                "visualization_type": visualization_type
            })
            
        return {"questions": plans}
    
    def _process_single_question(self, question: str, files: dict, analysis_plan: dict, question_index: int) -> dict:
        """Process a single question"""
        # Get the plan for this question
        plan = None
        for q_plan in analysis_plan.get("questions", []):
            if q_plan.get("index") == question_index:
                plan = q_plan
                break
        
        if not plan:
            plan = {
                "task_type": "data_analysis",
                "data_source": "uploaded_files",
                "analysis_method": "descriptive",
                "visualization_needed": True,
                "visualization_type": "scatter"
            }
        
        result = {"question": question}
        
        # Gather data based on plan
        data = self._gather_data(plan, files)
        
        if not data:
            return {"question": question, "error": "No data available for analysis"}
        
        # Perform analysis
        analysis_result = self._perform_data_analysis(data, plan, question)
        result.update(analysis_result)
        
        # Generate visualization if needed
        if plan.get("visualization_needed", False):
            try:
                viz_result = self._generate_visualization(data, plan, question)
                result.update(viz_result)
            except Exception as e:
                logger.warning(f"Visualization generation failed: {str(e)}")
                result["visualization_error"] = str(e)
        
        return result
    
    def _gather_data(self, plan: dict, files: dict) -> dict:
        """Gather data based on analysis plan"""
        data = {}
        
        if plan.get("data_source") in ["uploaded_files", "both"]:
            # Use uploaded files
            for filename, file_data in files.items():
                if isinstance(file_data, pd.DataFrame):
                    data[filename] = file_data
                    
        if plan.get("data_source") in ["web_scraping", "both"]:
            # Perform web scraping if needed
            try:
                scraped_data = self._perform_web_scraping(plan)
                if scraped_data:
                    data.update(scraped_data)
            except Exception as e:
                logger.warning(f"Web scraping failed: {str(e)}")
        
        return data
    
    def _perform_web_scraping(self, plan: dict) -> dict:
        """Perform web scraping based on plan"""
        # This is a simplified implementation - in practice, you'd use the plan to determine what to scrape
        # For demonstration, we'll scrape a common data source
        try:
            # Example: scrape some financial data or news
            url = "https://httpbin.org/json"  # Fallback URL for testing
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.json_normalize(data)
                return {"scraped_data": df}
        except Exception as e:
            logger.warning(f"Web scraping failed: {str(e)}")
        
        return {}
    
    def _perform_data_analysis(self, data: dict, plan: dict, question: str) -> dict:
        """Perform data analysis based on plan and question"""
        if not data:
            return {"answer": "No data available for analysis"}
        
        # Combine all DataFrames
        combined_df = None
        for name, df in data.items():
            if isinstance(df, pd.DataFrame):
                if combined_df is None:
                    combined_df = df.copy()
                    combined_df['source'] = name
                else:
                    df_copy = df.copy()
                    df_copy['source'] = name
                    combined_df = pd.concat([combined_df, df_copy], ignore_index=True)
        
        if combined_df is None or combined_df.empty:
            return {"answer": "No valid data found for analysis"}
        
        # Use intelligent analysis without requiring OpenAI API
        analysis_result = self._get_intelligent_analysis(combined_df, question, plan)
        
        return analysis_result
    
    def _get_intelligent_analysis(self, df: pd.DataFrame, question: str, plan: dict) -> dict:
        """Perform intelligent data analysis without requiring OpenAI API"""
        try:
            analysis_method = plan.get("analysis_method", "descriptive")
            numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
            categorical_columns = list(df.select_dtypes(include=['object']).columns)
            
            result = {
                "answer": "",
                "insights": [],
                "calculations": {},
                "supporting_data": {}
            }
            
            # Basic statistics
            if numeric_columns:
                stats = {}
                for col in numeric_columns:
                    try:
                        stats[col] = {
                            "mean": round(float(df[col].mean()), 3),
                            "std": round(float(df[col].std()), 3),
                            "min": round(float(df[col].min()), 3),
                            "max": round(float(df[col].max()), 3),
                            "count": int(df[col].count())
                        }
                    except:
                        continue
                result["calculations"]["statistics"] = stats
            
            # Perform specific analysis based on method
            if analysis_method == "correlation" and len(numeric_columns) >= 2:
                corr_matrix = df[numeric_columns].corr()
                # Find strongest correlations
                correlations = []
                for i in range(len(numeric_columns)):
                    for j in range(i+1, len(numeric_columns)):
                        col1, col2 = numeric_columns[i], numeric_columns[j]
                        corr_value = corr_matrix.loc[col1, col2]
                        if not np.isnan(corr_value):
                            correlations.append({
                                "variables": f"{col1} vs {col2}",
                                "correlation": round(float(corr_value), 3),
                                "strength": self._interpret_correlation(corr_value)
                            })
                
                correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                result["calculations"]["correlations"] = correlations[:5]  # Top 5
                
                if correlations:
                    strongest = correlations[0]
                    result["answer"] = f"The strongest correlation is between {strongest['variables']} with a coefficient of {strongest['correlation']} ({strongest['strength']})"
                    result["insights"].append(f"Found {len(correlations)} significant correlations in the data")
                else:
                    result["answer"] = "No significant correlations found in the numeric data"
                    
            elif analysis_method == "regression" and len(numeric_columns) >= 2:
                # Perform regression analysis
                target_col = numeric_columns[-1]  # Last column as target
                feature_cols = numeric_columns[:-1]
                
                if feature_cols:
                    try:
                        X = df[feature_cols].dropna()
                        y = df[target_col].dropna()
                        
                        # Align the data
                        common_idx = X.index.intersection(y.index)
                        X_aligned = X.loc[common_idx]
                        y_aligned = y.loc[common_idx]
                        
                        if len(X_aligned) > 1:
                            model = LinearRegression()
                            model.fit(X_aligned, y_aligned)
                            r2 = r2_score(y_aligned, model.predict(X_aligned))
                            
                            result["calculations"]["regression"] = {
                                "target": target_col,
                                "features": feature_cols,
                                "r_squared": round(float(r2), 3),
                                "coefficients": {col: round(float(coef), 3) for col, coef in zip(feature_cols, model.coef_)}
                            }
                            
                            result["answer"] = f"Regression model predicting {target_col} has R² = {round(r2, 3)}"
                            result["insights"].append(f"Model explains {round(r2*100, 1)}% of variance in {target_col}")
                    except Exception as e:
                        result["answer"] = f"Regression analysis failed: {str(e)}"
                        
            elif analysis_method == "comparison" and categorical_columns and numeric_columns:
                # Compare numeric values across categories
                cat_col = categorical_columns[0]
                num_col = numeric_columns[0]
                
                comparison_df = df.groupby(cat_col)[num_col].agg(['mean', 'count', 'std']).round(3)
                result["calculations"]["comparison"] = comparison_df.to_dict()
                
                max_category = comparison_df['mean'].idxmax()
                min_category = comparison_df['mean'].idxmin()
                
                result["answer"] = f"Highest average {num_col}: {max_category} ({comparison_df.loc[max_category, 'mean']}), Lowest: {min_category} ({comparison_df.loc[min_category, 'mean']})"
                result["insights"].append(f"Analyzed {len(comparison_df)} categories in {cat_col}")
                
            else:
                # Descriptive analysis
                result["answer"] = f"Dataset contains {len(df)} rows and {len(df.columns)} columns"
                result["insights"].append(f"Found {len(numeric_columns)} numeric and {len(categorical_columns)} categorical columns")
                
                if numeric_columns:
                    result["insights"].append(f"Numeric columns: {', '.join(numeric_columns[:3])}{'...' if len(numeric_columns) > 3 else ''}")
                if categorical_columns:
                    result["insights"].append(f"Categorical columns: {', '.join(categorical_columns[:3])}{'...' if len(categorical_columns) > 3 else ''}")
            
            # Add data quality insights
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                result["insights"].append(f"Data has {missing_data.sum()} missing values across {(missing_data > 0).sum()} columns")
            
            result["supporting_data"]["shape"] = df.shape
            result["supporting_data"]["data_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {"answer": f"Analysis failed: {str(e)}", "insights": [], "calculations": {}, "supporting_data": {}}
    
    def _interpret_correlation(self, corr_value):
        """Interpret correlation coefficient strength"""
        abs_corr = abs(corr_value)
        if abs_corr >= 0.8:
            return "very strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very weak"
    
    def _generate_visualization(self, data: dict, plan: dict, question: str) -> dict:
        """Generate visualization based on data and plan"""
        if not data:
            return {"visualization_error": "No data for visualization"}
        
        # Combine all DataFrames
        combined_df = None
        for name, df in data.items():
            if isinstance(df, pd.DataFrame):
                if combined_df is None:
                    combined_df = df.copy()
                else:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        if combined_df is None or combined_df.empty:
            return {"visualization_error": "No valid data for visualization"}
        
        # Get numeric columns
        numeric_cols = list(combined_df.select_dtypes(include=[np.number]).columns)
        
        if len(numeric_cols) < 1:
            return {"visualization_error": "No numeric columns found for visualization"}
        
        try:
            plt.figure(figsize=(10, 6))
            plt.style.use('default')
            
            viz_type = plan.get("visualization_type", "scatter")
            
            if viz_type == "scatter" and len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                
                plt.scatter(combined_df[x_col], combined_df[y_col], alpha=0.6)
                
                # Add regression line if requested
                if len(combined_df) > 1:
                    x_vals = combined_df[x_col].dropna()
                    y_vals = combined_df[y_col].dropna()
                    
                    if len(x_vals) > 1 and len(y_vals) > 1:
                        # Align the data
                        valid_indices = combined_df[[x_col, y_col]].dropna().index
                        x_clean = combined_df.loc[valid_indices, x_col].values.reshape(-1, 1)
                        y_clean = combined_df.loc[valid_indices, y_col].values
                        
                        if len(x_clean) > 1:
                            model = LinearRegression()
                            model.fit(x_clean, y_clean)
                            y_pred = model.predict(x_clean)
                            
                            # Sort for plotting line
                            sort_idx = np.argsort(x_clean.flatten())
                            plt.plot(x_clean[sort_idx], y_pred[sort_idx], 
                                   color='red', linestyle='--', linewidth=2, 
                                   label=f'Regression Line (R²={r2_score(y_clean, y_pred):.3f})')
                            plt.legend()
                
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f'Scatter Plot: {x_col} vs {y_col}')
                
            elif viz_type == "histogram" or len(numeric_cols) == 1:
                col = numeric_cols[0]
                plt.hist(combined_df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {col}')
                
            elif viz_type == "bar":
                # For bar charts, try to find a categorical column
                cat_cols = list(combined_df.select_dtypes(include=['object']).columns)
                if cat_cols and numeric_cols:
                    cat_col = cat_cols[0]
                    num_col = numeric_cols[0]
                    
                    grouped = combined_df.groupby(cat_col)[num_col].mean().head(10)  # Top 10 categories
                    grouped.plot(kind='bar')
                    plt.xlabel(cat_col)
                    plt.ylabel(f'Average {num_col}')
                    plt.title(f'Bar Chart: Average {num_col} by {cat_col}')
                    plt.xticks(rotation=45)
                else:
                    # Fallback to histogram
                    col = numeric_cols[0]
                    plt.hist(combined_df[col].dropna(), bins=30, alpha=0.7)
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.title(f'Histogram of {col}')
            
            else:
                # Default: line plot of first numeric column
                col = numeric_cols[0]
                plt.plot(combined_df[col].dropna())
                plt.ylabel(col)
                plt.xlabel('Index')
                plt.title(f'Line Plot of {col}')
            
            plt.tight_layout()
            
            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            
            # Check size
            img_size = len(buf.getvalue())
            if img_size > 100 * 1024:  # 100KB
                # Reduce DPI if too large
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=75, bbox_inches="tight")
                buf.seek(0)
            
            encoded = base64.b64encode(buf.read()).decode("utf-8")
            image_uri = f"data:image/png;base64,{encoded}"
            
            plt.close()
            
            return {"visualization": image_uri}
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            return {"visualization_error": str(e)}
