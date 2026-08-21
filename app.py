from fastapi import FastAPI
import boto3
import json
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AWS Learning Backend"}

@app.get("/health")
def health():
    return {"status": "ok"}


sagemaker_runtime = boto3.client("sagemaker-runtime", region_name="us-east-2")
SAGEMAKER_ENDPOINT_NAME = "aws-learning-iris-endpoint"

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisFeatures):
    payload = {
        "instances": [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]]
    }
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read())
    return {"prediction": result["predictions"][0]}