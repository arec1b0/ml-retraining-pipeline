"""
Test suite for M3 and M2 refactoring changes:
- M3: Move flow-specific tasks out of flows.py
- M2: API error handling returns generic messages
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestM3Refactoring:
    """Tests for M3: Task movement refactoring"""
    
    def test_load_reference_data_imported_from_data_module(self):
        """Verify load_reference_data is available from data.py"""
        from src.pipeline.tasks.data import load_reference_data
        assert callable(load_reference_data)
        assert hasattr(load_reference_data, '__name__')
        assert 'load_reference_data' in load_reference_data.__name__
    
    def test_simulate_current_data_imported_from_data_module(self):
        """Verify simulate_current_data is available from data.py"""
        from src.pipeline.tasks.data import simulate_current_data
        assert callable(simulate_current_data)
        assert hasattr(simulate_current_data, '__name__')
        assert 'simulate_current_data' in simulate_current_data.__name__
    
    def test_no_duplicate_imports_in_flows(self):
        """Verify flows.py imports tasks from data module, no duplicates"""
        import inspect
        from src.pipeline import flows
        
        # Get source code
        source = inspect.getsource(flows)
        
        # Check that imports come from data module
        assert 'from src.pipeline.tasks.data import' in source
        assert 'load_reference_data' in source
        assert 'simulate_current_data' in source
    
    def test_tasks_are_cohesively_grouped(self):
        """Verify all data tasks are in the data module"""
        from src.pipeline.tasks import data
        
        # Check that all data-related tasks are present
        data_tasks = [
            'load_raw_data',
            'validate_data',
            'preprocess_data',
            'split_data',
            'load_reference_data',
            'simulate_current_data',
        ]
        
        for task_name in data_tasks:
            assert hasattr(data, task_name), f"{task_name} not found in data module"
            assert callable(getattr(data, task_name))


class TestM2ErrorHandling:
    """Tests for M2: API error handling with generic messages"""
    
    def setup_method(self):
        """Setup test client before each test"""
        from inference_service.app.main import app
        self.client = TestClient(app)
        # Create a custom logger to capture logs
        self.log_handler = logging.handlers.MemoryHandler(capacity=1000)
        logger = logging.getLogger("inference_service.app.main")
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.ERROR)
    
    def test_generic_error_message_on_unhandled_exception(self):
        """Verify unhandled exceptions return generic error message"""
        # Mock model manager to raise an exception
        with patch('inference_service.app.main.model_manager') as mock_manager:
            mock_manager.is_loaded.return_value = True
            mock_manager.get_model_info.side_effect = RuntimeError("Secret DB connection string exposed!")
            
            response = self.client.get("/models/info")
            
            # Verify response status
            assert response.status_code == 500
            
            # Verify generic error message (no sensitive details)
            data = response.json()
            assert 'error' in data
            assert "Failed to retrieve model information" in data['detail']
            
            # Verify exception details are NOT in the response
            assert "Secret" not in str(response.text)
            assert "DB connection" not in str(response.text)
            assert "RuntimeError" not in str(response.text)
    
    def test_predict_endpoint_returns_generic_error(self):
        """Verify /predict returns generic error on failure"""
        with patch('inference_service.app.main.model_manager') as mock_manager:
            mock_manager.is_loaded.return_value = True
            mock_manager.predict.side_effect = Exception("Internal auth token: xyz123secret!")
            
            response = self.client.post(
                "/predict",
                json={"text": "Test text"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "Prediction failed" in data['detail']
            assert "auth token" not in str(response.text).lower()
            assert "secret" not in str(response.text).lower()
    
    def test_batch_predict_endpoint_returns_generic_error(self):
        """Verify /predict_batch returns generic error on failure"""
        with patch('inference_service.app.main.model_manager') as mock_manager:
            mock_manager.is_loaded.return_value = True
            mock_manager.predict.side_effect = ValueError("Database password: mySecret123!")
            
            response = self.client.post(
                "/predict_batch",
                json={"texts": ["text1", "text2"]}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "Batch prediction failed" in data['detail']
            assert "password" not in str(response.text).lower()
            assert "mysecret" not in str(response.text).lower()
    
    def test_error_logged_internally_with_details(self, caplog):
        """Verify detailed errors ARE logged internally"""
        with patch('inference_service.app.main.model_manager') as mock_manager:
            mock_manager.is_loaded.return_value = True
            mock_manager.predict.side_effect = Exception("INTERNAL_SECRET_KEY_12345")
            
            with caplog.at_level(logging.ERROR):
                response = self.client.post(
                    "/predict",
                    json={"text": "Test"}
                )
            
            # Verify the detailed error is logged
            assert any("INTERNAL_SECRET_KEY_12345" in record.message for record in caplog.records)
            
            # But NOT in the response
            assert "INTERNAL_SECRET_KEY_12345" not in response.json()['detail']
    
    def test_model_info_endpoint_generic_error(self):
        """Verify /models/info returns generic error"""
        with patch('inference_service.app.main.model_manager') as mock_manager:
            mock_manager.is_loaded.return_value = True
            mock_manager.get_model_info.side_effect = IOError("Failed to connect to mongodb://admin:pass@host:27017")
            
            response = self.client.get("/models/info")
            
            assert response.status_code == 500
            data = response.json()
            assert "Failed to retrieve model information" in data['detail']
            assert "mongodb" not in str(response.text).lower()
            assert "admin:pass" not in str(response.text)


class TestRefactoringIntegration:
    """Integration tests for the refactoring"""
    
    def test_flow_imports_work_correctly(self):
        """Verify the refactored flow imports work"""
        from src.pipeline.flows import retraining_flow
        from src.pipeline.tasks.data import (
            load_raw_data,
            load_reference_data,
            simulate_current_data,
        )
        
        # All should be callable
        assert callable(retraining_flow)
        assert callable(load_raw_data)
        assert callable(load_reference_data)
        assert callable(simulate_current_data)
    
    def test_no_circular_imports(self):
        """Verify no circular import issues"""
        try:
            from src.pipeline import flows
            from src.pipeline.tasks import data
            from inference_service.app import main
            
            # If we get here, no circular imports
            assert True
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")


class TestErrorMessageSecurity:
    """Security tests for error messages"""
    
    def test_no_stack_traces_in_api_response(self):
        """Verify stack traces are not exposed in API responses"""
        from inference_service.app.main import app
        client = TestClient(app)
        
        with patch('inference_service.app.main.model_manager') as mock_manager:
            mock_manager.is_loaded.return_value = True
            mock_manager.predict.side_effect = Exception("line 42: raise ValueError('test')")
            
            response = client.post(
                "/predict",
                json={"text": "test"}
            )
            
            response_text = str(response.json())
            # No "line 42", "File", or "Traceback" in response
            assert "line 42" not in response_text
            assert "Traceback" not in response_text
            assert "File " not in response_text
    
    def test_no_system_info_in_response(self):
        """Verify system information is not exposed"""
        from inference_service.app.main import app
        client = TestClient(app)
        
        with patch('inference_service.app.main.model_manager') as mock_manager:
            mock_manager.is_loaded.return_value = True
            mock_manager.predict.side_effect = Exception("/home/user/.ssh/id_rsa")
            
            response = client.post(
                "/predict",
                json={"text": "test"}
            )
            
            response_text = str(response.json())
            assert "/home/user" not in response_text
            assert ".ssh" not in response_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
