import joblib
import os
import json
import numpy as np

def model_fn(model_dir):
    # SageMaker llama esta función una vez, al arrancar, para cargar el modelo
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def input_fn(request_body, request_content_type):
    # Convierte el JSON que le mandemos desde FastAPI en algo que el modelo entienda
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["instances"])
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, response_content_type):
    # Convierte el resultado del modelo de vuelta a JSON
    return json.dumps({"predictions": prediction.tolist()})