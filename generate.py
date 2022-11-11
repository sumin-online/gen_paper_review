import json
from pathlib import Path
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

import torch

from hparams import Hyperparameter

def generate_tldr(abstract, model, tokenizer, max_length=15):
  model.eval()

  abstract_ids = tokenizer.encode(abstract)[1:-1]
  abstract_ids_truncated = abstract_ids[:int(0.9 * 512)]
  tldr_str = ""
  tldr_ids = []
  while True:
    tldr_input_str = tldr_str + " [MASK]"
    tldr_input_ids = tokenizer.encode(tldr_input_str)[1:-1]
    input_ids_list = [tokenizer.cls_token_id] + abstract_ids_truncated + [tokenizer.sep_token_id] + tldr_input_ids
    input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)
    if not tldr_ids:
      print("Abstract input: ", tokenizer.decode(input_ids_list))
    
    with torch.no_grad():
      output_logits = model(input_ids).logits
    mask_logits = output_logits[0, -1, :]
    top_idx = torch.argmax(mask_logits).cpu().item()

    if top_idx == tokenizer.sep_token_id:
      break

    tldr_ids.append(top_idx)
    answer_str = tokenizer.decode(tldr_ids)

    if len(tldr_ids) > max_length:
      print(tldr_ids)
      print(answer_str)
      break

  return answer_str

if __name__ == "__main__":
  import pickle
  import json

  with open("crawled/NeurIPS_2021.pkl", "rb") as f:
    paper_data = pickle.load(f)

  load_dir = Path("checkpoints/bert-base-uncased")

  with open(load_dir / "hp.json") as f:
    hp_dict = json.load(f)
  hparams = Hyperparameter(**hp_dict)
  device = torch.device(hparams.device)

  tokenizer = BertTokenizer.from_pretrained(hparams.pretrained_model)

  config = BertConfig.from_pretrained(load_dir)
  model = BertForMaskedLM.from_pretrained(load_dir, config=config).to(device)

  for idx in [0, 1, 2, 3]:
    abstract = paper_data[idx]["content"]["abstract"]
  
    generate_tldr(abstract, model, tokenizer, max_length=50)