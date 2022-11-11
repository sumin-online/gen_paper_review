from torch.utils.data import DataLoader
from transformers import BertConfig, BertForMaskedLM, AdamW
from tqdm import tqdm
from pathlib import Path

import torch
import json

from hparams import Hyperparameter
from preprocess import Preprocessor


def validate(model, loader, device):
  model.eval()
  total_loss = 0
  data_size = 0
  for batch in loader:
    _, input_ids, label_ids = batch
    input_ids = input_ids.to(device)
    label_ids = label_ids.to(device)
    
    with torch.no_grad():
      output = model(
        input_ids=input_ids, 
        labels=label_ids, 
        return_dict=True
      )
      loss = output.loss

    total_loss += loss.item() * len(input_ids)
    data_size += len(input_ids)
  
  avg_loss = total_loss / data_size
  return avg_loss


def train():
  # Hyperparameters
  hp = Hyperparameter()
  device = torch.device(hp.device)
  save_dir = Path(hp.save_dir) / hp.pretrained_model

  # Preprocessor
  config = BertConfig.from_pretrained(hp.pretrained_model)
  preprocessor = Preprocessor(hp, config=config)
  
  # train/dev/test dataset
  train_dataset, dev_dataset, test_dataset = preprocessor()

  train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=preprocessor.collate_fn)
  dev_loader = DataLoader(dev_dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=preprocessor.collate_fn)
  test_loader = DataLoader(test_dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=preprocessor.collate_fn)

  # Model
  model = BertForMaskedLM.from_pretrained(hp.pretrained_model, config=config).to(device)

  # Optimizer
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 1e-4
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0
    },
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=hp.learning_rate, eps=1e-8)

  min_dev_loss = float('inf')
  global_step = 0
  patience = 0
  early_stopping = False
  for epoch in range(hp.max_epochs):
    model.train()
    for batch in train_loader:
      _, input_ids, label_ids = batch
      input_ids = input_ids.to(device)
      label_ids = label_ids.to(device)
      
      output = model(
        input_ids=input_ids, 
        labels=label_ids, 
        return_dict=True
      )
      loss = output.loss
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      global_step += 1

      if global_step % hp.validation_steps == 0:

        dev_loss = validate(model, dev_loader, device)
        print(f'Step {global_step} | Dev loss: {dev_loss:.6f}')

        if dev_loss < min_dev_loss:
          min_dev_loss = dev_loss

          # Save model
          model.save_pretrained(save_dir)
          config.save_pretrained(save_dir)
          with open(save_dir / "hp.json", "w") as f:
            json.dump(hp.dict(), f, indent=4, ensure_ascii=False)

          status = {
            "global_step": global_step, 
            "epoch": epoch
          }
          with open(save_dir / "status.json", "w") as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
          
          # Notice model saved
          print(f"Model saved at step {global_step} / epoch {epoch}")
          patience = 0

        else:
          patience += hp.validation_steps
          if patience > hp.patience_steps:
            print(f"Early stopping: No improvement in {patience} epochs")
            early_stopping = True
            break
    
    if early_stopping:
      break

  # Test
  config = BertConfig.from_pretrained(save_dir)
  model = BertForMaskedLM.from_pretrained(save_dir, config=config).to(device)
  test_loss = validate(model, test_loader, device)
  print(f'Test Loss: {test_loss:.6f}')

if __name__ == '__main__':
  train()
