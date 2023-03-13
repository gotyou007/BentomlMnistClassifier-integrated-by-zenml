import click
from constants import MODEL_NAME, PIPELINE_NAME, PIPELINE_STEP_NAME
from pipelines.inference_fashion_mnist import inference_fashion_mnist
from pipelines.training_fashion_mnist import training_fashion_mnist
from steps.bento_builder import bento_builder
from steps.deployer import bentoml_model_deployer
from steps.deployment_trigger_step import (
    DeploymentTriggerParameters,
    deployment_trigger,
)
from steps.evaluators import evaluator
from steps.importers import importer_mnist
from steps.inference_loader import inference_loader
from steps.prediction_service_loader import (
    PredictionServiceLoaderStepParameters,
    bentoml_prediction_service_loader,
)
from steps.predictor import predictor
from steps.trainers import trainer

import requests

requests.post(
   "http://127.0.0.1:3001/predict_ndarray",
   headers={"content-type": "application/json"},
   data="images['sneaker.jpeg']",
).text

import bentoml

mnist_runner = bentoml.pytorch.get("pytorch_mnist").to_runner()
mnist_runner.init_local()
mnist_runner.predict.run(images['sneaker.jpeg'])

curl -H "Content-Type: multipart/form-data" -F'fileobj=@inference_samples/ankle_boot.jpg;type=image/png' http://127.0.0.1:3001/predict_image