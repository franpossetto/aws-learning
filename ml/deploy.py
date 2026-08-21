import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig

# Pineamos el bucket default del SDK al nuestro, donde el execution role
# ya tiene permiso de lectura -- si no, el SDK usaría otro bucket que
# el role no puede leer, y volveríamos a fallar por permisos.
boto_sess = boto3.Session(region_name="us-east-2")
sm_session = sagemaker.Session(
    boto_session=boto_sess,
    default_bucket="aws-learning-sagemaker-models-637423212230-us-east-2-an",
)

role = "arn:aws:iam::637423212230:role/aws-learning-sagemaker-execution-role"
model_data = "s3://aws-learning-sagemaker-models-637423212230-us-east-2-an/model-artifact.tar.gz"

sklearn_model = SKLearnModel(
    model_data=model_data,
    role=role,
    entry_point="inference.py",
    source_dir="code",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sm_session,
)

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=1024,
    max_concurrency=20,
)

predictor = sklearn_model.deploy(
    endpoint_name="aws-learning-iris-endpoint",
    serverless_inference_config=serverless_config,
)

print("Endpoint desplegado:", predictor.endpoint_name)