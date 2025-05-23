from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import io

from app.main import app

@patch('ultralytics.YOLO')
def test_root_endpoint(mock_yolo):
    """Тест корневого эндпоинта"""
    # Настраиваем мок
    mock_model = Mock()
    mock_yolo.return_value = mock_model
    
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "YOLO Object Detection API работает" in response.json()["message"]

@patch('ultralytics.YOLO')
def test_health_check(mock_yolo):
    """Тест проверки здоровья"""
    mock_model = Mock()
    mock_yolo.return_value = mock_model
    
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

@patch('ultralytics.YOLO')
def test_detect_invalid_file(mock_yolo):
    """Тест с неправильным типом файла"""
    mock_model = Mock()
    mock_yolo.return_value = mock_model
    
    with TestClient(app) as client:
        text_content = b"This is not an image"
        files = {"file": ("test.txt", io.BytesIO(text_content), "text/plain")}
        
        response = client.post("/detect", files=files)
        assert response.status_code == 400
        assert "изображением" in response.json()["detail"]

@patch('ultralytics.YOLO')
def test_detect_base64_missing_image(mock_yolo):
    """Тест без поля image"""
    mock_model = Mock()
    mock_yolo.return_value = mock_model
    
    with TestClient(app) as client:
        payload = {"confidence": 0.5}
        response = client.post("/detect_base64", json=payload)
        assert response.status_code == 400
        assert "обязательно" in response.json()["detail"]

@patch('ultralytics.YOLO')
def test_detect_endpoint_exists(mock_yolo):
    """Проверяем, что эндпоинт детекции существует"""
    mock_model = Mock()
    mock_yolo.return_value = mock_model
    
    with TestClient(app) as client:
        # Без файла должна быть ошибка 422
        response = client.post("/detect")
        assert response.status_code == 422

@patch('ultralytics.YOLO')
def test_detect_base64_endpoint_exists(mock_yolo):
    """Проверяем, что эндпоинт base64 детекции существует"""
    mock_model = Mock()
    mock_yolo.return_value = mock_model
    
    with TestClient(app) as client:
        # Без данных должна быть ошибка 422
        response = client.post("/detect_base64", json={})
        assert response.status_code in [400, 422]
