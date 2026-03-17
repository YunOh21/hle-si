from flask import Flask, request, jsonify
from sensenova_si import get_model
import os

app = Flask(__name__)

print("Loading SenseNova-SI model...")
model_path = "sensenova/SenseNova-SI-1.3-InternVL3-8B"
local_model = get_model(model_path, model_type="auto")
print("Model loaded and ready!")

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    prompt = data.get('prompt')
    image_paths = data.get('image_paths')
    
    try:
        res = local_model.generate(prompt, images=image_paths)
        return jsonify({"status": "success", "result": res})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)