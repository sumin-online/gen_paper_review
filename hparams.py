from pydantic import BaseModel
import torch


class Hyperparameter(BaseModel):
    # Device
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Model & Optimizer
    pretrained_model: str = "bert-base-uncased"
    learning_rate: float = 5e-5

    # Data Loading
    batch_size: int = 8

    # Training
    max_epochs: int = 20
    save_dir: str = "checkpoints/"
    validation_steps: int = 200
    patience_steps: int = 2000

    # WikiQA
    wikiqa_document_path: str = "data/wikiQA/wikiQA-documents.json"
