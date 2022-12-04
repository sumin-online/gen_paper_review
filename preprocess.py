from dataclasses import dataclass
import pickle
import random
from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import BertConfig, BertTokenizer

from hparams import Hyperparameter


@dataclass
class PaperAssessExample:
    abstract: str
    tldr: str
    accepted: bool

@dataclass
class PaperTarget:
    abstract: str
    target: bool or str

@dataclass
class PaperAssessFeature:
    input_ids: List[int]
    label_ids: List[int]


class PaperAssessDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self, features: List[PaperAssessFeature]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> PaperAssessFeature:
        return self.features[idx]


class Preprocessor:
    def __init__(self, hp: Hyperparameter, config: Any):
        self.hp = hp
        self.tokenizer = BertTokenizer.from_pretrained(self.hp.pretrained_model)
        self.config = config
        self.pad_id = self.tokenizer.pad_token_id
        self.max_seq_len = self.config.max_position_embeddings # 512

    def read_example(self, raw_data, task):
        all_data = []
        for raw_datum in raw_data:
            # 1) content
            content = raw_datum.get("content")
            if content is None:
                continue

            # 2) abstract
            abstract = content.get("abstract")
            if abstract is None:
                continue

            # 3) target
            if task == "tldr":
                target = content.get("TL;DR")
                    
            elif task == "accepted":
                target = raw_datum.get("accepted")
            
            if target is None:
                continue

            # 4) save
            datapoint = PaperTarget(abstract = abstract,
                                    target = target)
            all_data.append(datapoint)

        return all_data                    
            

    def read_tldr(self, raw_data) -> List[PaperAssessExample]:


        all_data = []
        for raw_datum in raw_data:
            content = raw_datum.get("content")
            if content is None:
                continue

            abstract = content.get("abstract")
            tldr = content.get("TL;DR")
            accepted = content.get("accepted")

            if tldr is None:
                continue

            datapoint = PaperAssessExample(abstract=abstract, 
                                           tldr=tldr,
                                           accepted=accepted)
            all_data.append(datapoint)

        return all_data

    def convert_examples_to_features(
        self, examples: List[PaperAssessExample], task = None
    ) -> List[PaperAssessFeature]:
        features = []
        for example in examples:
            document = example.abstract
            target = example.target

            # Preprocess (If needed)

            abstract_ids = self.tokenizer.encode(document)[1:-1]
            abstract_ids_truncated = abstract_ids[: int(0.9 * self.max_seq_len)]
            abstract_ids_input = (
                [self.tokenizer.cls_token_id]
                + abstract_ids_truncated
                + [self.tokenizer.sep_token_id]
            )

            if task == "accepted":
                features.append(
                    PaperAssessFeature(
                        input_ids = abstract_ids_input,
                        label_ids = int(target),
                    )
                )
                
            else:
                target_ids = self.tokenizer.encode(target)[1:-1] + [self.tokenizer.sep_token_id]
                for i in range(len(target_ids)):
                    answer_input = target_ids[: i + 1]
                    answer_label = answer_input[i]
                    answer_input[i] = self.tokenizer.mask_token_id
                    input_ids = abstract_ids_input + answer_input
                    label_ids = [-100] * (len(input_ids) - 1) + [answer_label]
                    if len(input_ids) > self.max_seq_len:
                        continue

                    features.append(
                        PaperAssessFeature(
                            input_ids=input_ids,
                            label_ids=label_ids,
                        )
                    )
        return features
    
    def read_pickle(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def read_data(self):
        paths = ["crawled/NeurIPS_2021.pkl", 
                 "crawled/ICLR_2021.pkl",
                 "crawled/ICLR_2022.pkl"]

        total_data = []
        for path in paths:
            total_data += self.read_pickle(path)
        
        return total_data

    def split_data(self, examples : list):
        random.shuffle(examples)
        split1, split2 = int(len(examples) * 0.8), int(len(examples) * 0.9)
        features = {
                "train": examples[:split1],
                "dev": examples[split1:split2],
                "test": examples[split2:],
                }
        return features

    def preprocess(
        self, task: str
    ) -> Tuple[PaperAssessDataset, PaperAssessDataset, PaperAssessDataset]:

        # Load dataset
        raw_data = self.read_data()
        examples = self.read_example(raw_data, task)
        print("Data reading complete")

        # Split train & val & test
        examples = self.convert_examples_to_features(examples, task)
        print("Tokenization complete")

        if task == "accepted":
            accept_true, accept_false = [], []
            for e in examples:
                if e.label_ids:
                    accept_true.append(e)
                else:
                    accept_false.append(e) 
            
            accept_true_split = self.split_data(accept_true)
            accept_false_split = self.split_data(accept_false)
            
            # integrate
            features = {}
            for i in ["train", "dev", "test"]:
                features[i] = accept_true_split[i] + accept_false_split[i]
        else:
            features = self.split_data(examples)
      
        # make dataset
        train_dataset = PaperAssessDataset(features["train"])
        dev_dataset = PaperAssessDataset(features["dev"])
        test_dataset = PaperAssessDataset(features["test"])

        return train_dataset, dev_dataset, test_dataset

    def __call__(
        self, task: str
    ) -> Tuple[PaperAssessDataset, PaperAssessDataset, PaperAssessDataset]: # train_dataset, dev_dataset, test_dataset 
        return self.preprocess(task)

    def _pad_ids(self, input_ids: List[List[int]], pad_id: int) -> List[List[int]]:
        # max_seq_len = max([len(x) for x in input_ids])
        padded_input_ids = []
        for ids in input_ids:
            if len(ids) > self.max_seq_len:
                padded_ids = ids[: self.max_seq_len]
            else:
                padded_ids = ids + [pad_id] * (self.max_seq_len - len(ids))

            padded_input_ids.append(padded_ids)

        return padded_input_ids

    def collate_fn(self, batch: List[PaperAssessFeature]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(
            self._pad_ids([b.input_ids for b in batch], self.pad_id), dtype=torch.long
        )
        label_id = torch.tensor(
            self._pad_ids([b.label_ids for b in batch], self.pad_id), dtype=torch.long
        )

        return input_ids, label_id

    def collate_fn_accept(self, batch: List[PaperAssessFeature]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(self._pad_ids([b.input_ids for b in batch], self.pad_id), dtype=torch.long)
        label_id  = torch.tensor([b.label_ids for b in batch])
        return input_ids, label_id



if __name__ == "__main__":
    preprocessor = Preprocessor(Hyperparameter(), BertConfig.from_pretrained("bert-base-uncased"))
    train, dev, test = preprocessor("strength")
    print(len(train))
