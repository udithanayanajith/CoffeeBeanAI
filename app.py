from flask import Flask, request, jsonify
from PIL import Image
import io
from flask_cors import CORS
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
import joblib
import pandas as pd
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
VERIFICATION_MODEL_PATH = "ai_models/beans_verification_model.pth"
CLASSIFICATION_MODEL_PATH = "ai_models/beans_classification_model.pth"

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load tabular model and mappings
tabular_model = joblib.load("ai_models/bean_predictor.pkl")
with open("ai_models/bean_label_mappings.json", "r") as f:
    mappings = json.load(f)

def decode(column, value):
    return mappings[column].get(str(value), "Unknown")

def load_model(model_path):
    """Load a single model from a specific path"""
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model
        num_classes = len(checkpoint['class_names'])
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
         
        return model, checkpoint['class_names']
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {str(e)}")

# Load models
try:
    verification_model, VERIFICATION_CLASSES = load_model(VERIFICATION_MODEL_PATH)
    classification_model, CLASSIFICATION_CLASSES = load_model(CLASSIFICATION_MODEL_PATH)
except Exception as e:
    print(f"Error loading models: {str(e)}")
    exit(1)

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not file:
        return jsonify({'error': 'Empty file provided'}), 400
    
    try:
        # Read image directly into memory
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Verification step
        with torch.no_grad():
            verification_outputs = verification_model(image_tensor)
            verification_probs = torch.nn.functional.softmax(verification_outputs.logits, dim=1)
            is_bean = torch.argmax(verification_probs, dim=1).item()
            bean_confidence = verification_probs[0][is_bean].item()
        
        result = {
            'is_coffee_bean': bool(is_bean),
            'is_coffee_bean_confidence': f"{bean_confidence*100:.2f}%",
            'bean_type_confidence': None,
            'predicted_class': None,
        }
        
        # Classification step if it's a coffee bean
        if is_bean:
            with torch.no_grad():
                classification_outputs = classification_model(image_tensor)
                classification_probs = torch.nn.functional.softmax(classification_outputs.logits, dim=1)
                pred_class = torch.argmax(classification_probs, dim=1).item()
                type_confidence = classification_probs[0][pred_class].item()
            
            result.update({
                'bean_type_confidence': f"{type_confidence*100:.2f}%",
                'predicted_class': CLASSIFICATION_CLASSES[pred_class]
            })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    data = request.get_json()

    input_df = pd.DataFrame([{
        "symptoms_lable": data.get("symptoms_lable"),
        "category": data.get("category"),
        "region": data.get("region"),
        "dehydration_Duration": data.get("dehydration_Duration"),
        "caught_Rain/Mist": data.get("caught_Rain/Mist")
    }])

    probas = tabular_model.predict_proba(input_df)
    preds = tabular_model.predict(input_df)[0]

    response = {
        "bean_Quality": {
            "prediction": decode("Impact on Bean Quality", preds[0]),
            "probability": f"{probas[0][0][preds[0]] * 100:.0f}%"
        },
        "cause_condition": {
            "prediction": decode("Cause/Condition", preds[1]),
            "probability": f"{probas[1][0][preds[1]] * 100:.0f}%"
        },
        "defect_Name": {
            "prediction": decode("Defect Name", preds[2]),
            "probability": f"{probas[2][0][preds[2]] * 100:.0f}%"
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False)
