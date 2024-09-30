from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

num_classes = 2  # Ganti dengan jumlah kelas Anda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inisialisasi model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('model_citra_medis.pth', map_location=device))
model.to(device)
model.eval()

# Transformasi input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)

    # Konversi hasil prediksi ke label kelas
    class_names = train_dataset.classes
    predicted_class = class_names[predicted.item()]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
