"""
Quick verification tests for M3 and M2 refactoring
These tests verify the refactoring without runtime dependencies
"""

import ast
import inspect
from pathlib import Path


class TestM3RefactoringStructure:
    """Verify M3 refactoring by analyzing code structure"""
    
    def test_load_reference_data_exists_in_data_module(self):
        """Verify load_reference_data task exists in data.py"""
        data_file = Path("src/pipeline/tasks/data.py")
        content = data_file.read_text()
        
        assert "def load_reference_data" in content
        assert "@task(name=\"Load Reference Data\")" in content
        assert "Loads the reference dataset for drift comparison" in content
    
    def test_simulate_current_data_exists_in_data_module(self):
        """Verify simulate_current_data task exists in data.py"""
        data_file = Path("src/pipeline/tasks/data.py")
        content = data_file.read_text()
        
        assert "def simulate_current_data" in content
        assert "@task(name=\"Simulate Current Data Generation\")" in content
        assert "Simulates a \"current\" dataset for drift analysis" in content
    
    def test_tasks_not_duplicated_in_flows(self):
        """Verify task definitions are NOT in flows.py (removed)"""
        flows_file = Path("src/pipeline/flows.py")
        content = flows_file.read_text()
        
        # These should NOT be task definitions in flows.py
        # They should only be imported
        lines = content.split('\n')
        
        # Count @task decorators
        task_defs = [l for l in lines if "@task(name=\"Load Reference Data\")" in l or 
                     "@task(name=\"Simulate Current Data Generation\")" in l]
        
        # Should have 0 task definitions (they're imported, not defined)
        assert len(task_defs) == 0, "Task definitions should be removed from flows.py"
    
    def test_flows_imports_from_data_module(self):
        """Verify flows.py imports tasks from data.py"""
        flows_file = Path("src/pipeline/flows.py")
        content = flows_file.read_text()
        
        # Should import from data module
        assert "from src.pipeline.tasks.data import" in content
        assert "load_reference_data" in content
        assert "simulate_current_data" in content
    
    def test_no_mlflow_client_import_in_flows(self):
        """Verify unused MlflowClient import was removed"""
        flows_file = Path("src/pipeline/flows.py")
        content = flows_file.read_text()
        
        # Should NOT have MlflowClient import
        assert "from mlflow.tracking import MlflowClient" not in content
    
    def test_mlflow_import_exists_in_flows(self):
        """Verify mlflow module is imported"""
        flows_file = Path("src/pipeline/flows.py")
        content = flows_file.read_text()
        
        # Should import mlflow
        assert "import mlflow" in content


class TestM2ErrorHandlingSecurity:
    """Verify M2 refactoring by analyzing error handling"""
    
    def test_generic_error_in_global_exception_handler(self):
        """Verify global exception handler returns generic error"""
        main_file = Path("inference-service/app/main.py")
        content = main_file.read_text()
        
        # Find the global exception handler
        assert "@app.exception_handler(Exception)" in content
        assert "content={\"error\": \"Internal server error\"}" in content
        
        # Should NOT return the exception string
        assert "detail\": str(exc)" not in content
    
    def test_predict_endpoint_generic_error(self):
        """Verify /predict endpoint returns generic error"""
        main_file = Path("inference-service/app/main.py")
        content = main_file.read_text()
        
        # Find predict error handling
        assert "detail=\"Prediction failed\"" in content
        
        # Check it's not exposing exception details
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'detail="Prediction failed"' in line:
                # Found the error line, verify it doesn't include str(e)
                context = '\n'.join(lines[max(0, i-3):min(len(lines), i+3)])
                assert "str(e)" not in context, "Exception details should not be exposed"
    
    def test_batch_predict_endpoint_generic_error(self):
        """Verify /predict_batch endpoint returns generic error"""
        main_file = Path("inference-service/app/main.py")
        content = main_file.read_text()
        
        # Find batch predict error handling
        assert "detail=\"Batch prediction failed\"" in content
    
    def test_model_info_endpoint_generic_error(self):
        """Verify /models/info endpoint returns generic error"""
        main_file = Path("inference-service/app/main.py")
        content = main_file.read_text()
        
        # Find model info error handling
        assert "detail=\"Failed to retrieve model information\"" in content
    
    def test_exc_info_added_to_logging(self):
        """Verify exc_info=True is used for detailed logging"""
        main_file = Path("inference-service/app/main.py")
        content = main_file.read_text()
        
        # Check that exc_info=True is used for full stack traces in logs
        assert "exc_info=True" in content
        assert content.count("exc_info=True") >= 3  # At least 3 occurrences (global + 2 endpoints)


class TestCodeQuality:
    """Verify code quality improvements"""
    
    def test_data_module_organized(self):
        """Verify data module is well-organized"""
        data_file = Path("src/pipeline/tasks/data.py")
        content = data_file.read_text()
        
        # Should have clear imports
        assert "from prefect import task" in content
        
        # Should have all data tasks
        tasks = [
            "load_raw_data",
            "validate_data",
            "preprocess_data",
            "split_data",
            "load_reference_data",
            "simulate_current_data",
        ]
        
        for task in tasks:
            assert f"def {task}" in content, f"Missing {task} function"
    
    def test_api_docstrings_improved(self):
        """Verify API error handlers have improved documentation"""
        main_file = Path("inference-service/app/main.py")
        content = main_file.read_text()
        
        # Check for improved documentation
        assert "Logs detailed error information internally" in content or \
               "logs detailed errors internally" in content
        assert "generic error message" in content


class TestNoRegressions:
    """Verify no regressions were introduced"""
    
    def test_flows_syntax_valid(self):
        """Verify flows.py has valid Python syntax"""
        flows_file = Path("src/pipeline/flows.py")
        content = flows_file.read_text()
        
        try:
            ast.parse(content)
            assert True
        except SyntaxError as e:
            assert False, f"flows.py has syntax error: {e}"
    
    def test_data_module_syntax_valid(self):
        """Verify data.py has valid Python syntax"""
        data_file = Path("src/pipeline/tasks/data.py")
        content = data_file.read_text()
        
        try:
            ast.parse(content)
            assert True
        except SyntaxError as e:
            assert False, f"data.py has syntax error: {e}"
    
    def test_main_app_syntax_valid(self):
        """Verify main.py has valid Python syntax"""
        main_file = Path("inference-service/app/main.py")
        content = main_file.read_text()
        
        try:
            ast.parse(content)
            assert True
        except SyntaxError as e:
            assert False, f"main.py has syntax error: {e}"
    
    def test_no_import_circular_dependencies(self):
        """Verify no circular import patterns"""
        flows_file = Path("src/pipeline/flows.py")
        content = flows_file.read_text()
        
        # flows.py should import from data, but data should not import from flows
        assert "from src.pipeline.tasks.data import" in content
        
        # Check data module doesn't import flows
        data_file = Path("src/pipeline/tasks/data.py")
        data_content = data_file.read_text()
        assert "from src.pipeline.flows import" not in data_content


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
