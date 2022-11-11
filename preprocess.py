from datasets import load_dataset
from transformers import GPT2Config, GPT2Tokenizer, BertConfig, BertTokenizer
from dataclasses import dataclass
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, List

import pickle
import torch
import random

@dataclass
class PaperAssessDatapoint:
  id: str
  abstract: str
  tldr: str
  accepted: bool

@dataclass
class PaperAssessFeature:
  id: str
  input_ids: List
  label_id: int

class PaperAssessDataset(Dataset):
  def __init__(self, features):
    self.features = features

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    return self.features[idx]

class Preprocessor:
  def __init__(self, hp, config):
    self.hp = hp
    self.tokenizer = BertTokenizer.from_pretrained(self.hp.pretrained_model)
    self.config = config
    self.pad_id = self.tokenizer.pad_token_id
    self.max_seq_len = self.config.max_position_embeddings
  
  def read_data(self):
    # Load dataset
    with open("crawled/NeurIPS_2021.pkl", "rb") as f:
      raw_data = pickle.load(f)
    
    all_data = []
    for i, raw_datum in enumerate(raw_data):
      content = raw_datum.get("content")
      if content is None:
        continue

      abstract = content.get("abstract")
      tldr = content.get("TL;DR")
      accepted = content.get("accepted")

      if tldr is None:
        continue

      datapoint = PaperAssessDatapoint(
        id=i, 
        abstract=abstract, 
        tldr=tldr, 
        accepted=accepted
      )
      all_data.append(datapoint)

    return all_data
  
  def convert_examples_to_features(self, examples):
    features = []
    feature_idx = 0
    for example in examples:
      document = example.abstract
      tldr = example.tldr

      # Preprocess
      # document = document.replace(".", "").replace("?", "")
      # question = question.replace(".", "").replace("?", "")
      # answer = answer.replace(".", "").replace("?", "")

      abstract_ids = self.tokenizer.encode(document)[1:-1]
      abstract_ids_truncated = abstract_ids[:int(0.9 * self.max_seq_len)]
      abstract_ids_input = [self.tokenizer.cls_token_id] + abstract_ids_truncated + [self.tokenizer.sep_token_id]
      tldr_ids = self.tokenizer.encode(tldr)[1:-1] + [self.tokenizer.sep_token_id]

      for i in range(len(tldr_ids)):
        answer_input = tldr_ids[:i + 1]
        answer_label = answer_input[i]
        answer_input[i] = self.tokenizer.mask_token_id
        input_ids = abstract_ids_input + answer_input
        label_ids = [-100] * (len(input_ids) - 1) + [answer_label]
        if len(input_ids) > self.max_seq_len:
          continue

        features.append(PaperAssessFeature(
          id=feature_idx, 
          input_ids=input_ids, 
          label_id=label_ids, 
        ))
        feature_idx += 1
    return features

  def preprocess(self): 
    examples = self.read_data()
    print('Data reading complete')
    random.shuffle(examples)
    split1, split2 = int(len(examples) * 0.8), int(len(examples) * 0.9)
    examples_split = {
      "train": examples[:split1], 
      "dev": examples[split1:split2], 
      "test": examples[split2:]
    }
    features = {split: self.convert_examples_to_features(ex) for split, ex in examples_split.items()}
    print('Tokenization complete')
    
    train_dataset = PaperAssessDataset(features['train'])
    dev_dataset = PaperAssessDataset(features['dev'])
    test_dataset = PaperAssessDataset(features['test'])
    return train_dataset, dev_dataset, test_dataset

  def __call__(self):
    return self.preprocess()

  def _pad_ids(self, input_ids, pad_id):
    # max_seq_len = max([len(x) for x in input_ids])
    padded_input_ids = []
    for ids in input_ids:
      if len(ids) > self.max_seq_len:
        padded_ids = ids[:self.max_seq_len]
      else:
        padded_ids = ids + [pad_id] * (self.max_seq_len - len(ids))
      
      padded_input_ids.append(padded_ids)
    
    return padded_input_ids

  def collate_fn(self, batch):
    ids = [b.id for b in batch]
    input_ids = torch.tensor(self._pad_ids([b.input_ids for b in batch], self.pad_id), dtype=torch.long)
    label_id = torch.tensor(self._pad_ids([b.label_id for b in batch], self.pad_id), dtype=torch.long)

    return ids, input_ids, label_id

if __name__ == '__main__':
  from hparams import Hyperparameter
  preprocessor = Preprocessor(Hyperparameter(), BertConfig.from_pretrained('bert-base-uncased'))
  train, dev, test = preprocessor()
  print(len(train))

