from flask import Flask, render_template, request, jsonify
import torch
import base64
import io
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# Função para carregar um modelo
def load_model(path='../split-learning/savemodels/client1.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    model.to(device)
    model.eval()
    return model    

# Carregar o modelo ao iniciar o aplicativo Flask
mymodel = load_model()

# Transformação para pré-processamento da imagem no formato CIFAR10
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter os dados da imagem enviada pelo usuário
        img_bytes = request.form['imageDataUrl']
        
        # Decodifica a imagem base64
        img_data = base64.b64decode(img_bytes)
        
        # Converte os dados em um objeto de imagem PIL
        img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # Aplica transformações para o formato CIFAR10
        img = transform(img_pil)
        img = img.unsqueeze(0)
        
        # Mova a imagem para o mesmo dispositivo que o modelo
        device = next(mymodel.parameters()).device  # Obtém o dispositivo do modelo
        img = img.to(device)
        
        # Verificar as dimensões da entrada antes de fazer a predição
        print(f'Input shape: {img.shape}')
        
        # Fazer a predição com o modelo carregado
        with torch.no_grad():
            prediction = mymodel(img)
            predicted_label = torch.argmax(prediction).item()
        
        return jsonify({'prediction': int(predicted_label)})
    
    except Exception as e:
        return jsonify({'error': str(e)})



if __name__ == '__main__':
    app.run(debug=True, port=7500)
