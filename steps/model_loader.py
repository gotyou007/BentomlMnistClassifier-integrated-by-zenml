import os
import sys
from typing import Dict
import torch
from zenml.post_execution import get_pipeline
from zenml.steps import Output, step


@step
def model_loader() -> Output(model=torch.nn.Module):
    """Loads the trained models from previous training pipeline runs."""
    training_pipeline = get_pipeline("training_fashion_mnist")
    last_run = training_pipeline.runs[0]
    model_path = "./inference/model/best.pt"

    try:
        model = last_run.get_step("trainer").output.read()
    except KeyError:
        print(
            f"Skipping {last_run.name} as it does not contain the trainer step"
        )
    model_saver(model, model_path)
    return model


def model_saver(model: Dict, model_path: str):
    """Saves the model to the local file system."""
    os.makedirs(os.path.join("inference", "model"), exist_ok=True)
    torch.save(model, model_path)
