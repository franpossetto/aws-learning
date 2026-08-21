import boto3
import json

client = boto3.client("sagemaker-runtime", region_name="us-east-2")

payload = {"instances": [[5.1, 3.5, 1.4, 0.2]]}

response = client.invoke_endpoint(
    EndpointName="aws-learning-iris-endpoint",
    ContentType="application/json",
    Body=json.dumps(payload),
)

result = json.loads(response["Body"].read())
print("Predicción:", result)