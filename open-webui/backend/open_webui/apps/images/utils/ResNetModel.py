from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
from open_webui.apps.images.utils import ResNetModel

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResNetModelInference:
    def __init__(self, model_path, num_classes=1000):
        # Menggunakan ResNetModel yang sudah diimpor dari open_webui
        try:
            self.model = ResNetModel(model_path=model_path, num_classes=num_classes)  # Memakai model yang diimpor
            self.model.eval()
            logger.info("Model berhasil dimuat.")
        except Exception as e:
            logger.exception(f"Gagal memuat model: {e}")
            raise e
        
        # Transformasi gambar agar sesuai dengan input yang diharapkan oleh ResNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def predict_image(self, image):
        try:
            img = image.convert('RGB')
            img = self.transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(img)
            confidence, predicted = torch.max(outputs, 1)
            # Mengembalikan prediksi dan confidence score
            return predicted.item(),
            
        except Exception as e:
            logger.exception(f"Error saat melakukan prediksi: {e}")
            raise e

