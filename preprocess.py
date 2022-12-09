from dataclasses import dataclass
import pickle
import random
from typing import Any, Dict, List, Tuple, Union

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
    target: Union[bool, str]


@dataclass
class PaperAssessFeature:
    input_ids: List[int]
    label_ids: Union[List[int], int]


class PaperAssessDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self, features: List[PaperAssessFeature]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> PaperAssessFeature:
        return self.features[idx]


class Preprocessor:
    def __init__(self, hp: Hyperparameter, config: Any, task: str):
        self.hp = hp
        self.tokenizer = BertTokenizer.from_pretrained(self.hp.pretrained_model)
        self.config = config
        self.pad_id = self.tokenizer.pad_token_id
        self.max_seq_len = self.config.max_position_embeddings  # 512
        self.task = task

    def read_example(self, raw_data: Any) -> List[PaperTarget]:
        """
        Read example from raw data to predefined class PaperTarget.

        :param raw_data: Collected data, raw.

        :return: List of PaperTarget.
        """
        all_data_list = []
        for content in raw_data:

            # 1) abstract
            abstract = content.get("abstract")
            if abstract is None:
                continue

            # 2) target
            if self.task == "tldr":
                target = content.get("tl_dr")

            elif self.task == "accepted":
                target = content.get("accepted")

            elif self.task == "weakness":
                target = content.get("weaknesses")

            elif self.task == "strength":
                target = content.get("strengths")

            else:
                raise ValueError("[Error] Task is not defined")

            if target is None:
                continue

            # 4) save
            datapoint = (abstract, target)
            all_data_list.append(datapoint)

        # remove duplication
        all_data_nodup = list(set(all_data_list))

        # type
        all_data = [PaperTarget(abstract=x, target=y) for x, y in all_data_nodup]
        return all_data

    def convert_examples_to_features(self, examples: List[PaperTarget]) -> List[PaperAssessFeature]:
        """
        Convert data examples into features which become the input of model.

        :param examples: List of PaperTargets from read_example

        :return: List of PaperAssessFeature.
        """
        features = []
        for example in examples:
            document = example.abstract
            target = example.target

            # Preprocess (If needed)

            abstract_ids = self.tokenizer.encode(document, truncation=True)[1:-1]
            abstract_ids_truncated = abstract_ids[: int(0.9 * self.max_seq_len)]
            abstract_ids_input = (
                [self.tokenizer.cls_token_id]
                + abstract_ids_truncated
                + [self.tokenizer.sep_token_id]
            )

            if self.task == "accepted":
                features.append(
                    PaperAssessFeature(
                        input_ids=abstract_ids_input,
                        label_ids=int(target),
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

    def read_data(self) -> Any:
        if self.task in ["tldr", "accepted"]:
            path = "crawled/reviews_without_weaknesses.pkl"
        elif self.task in ["weakness", "strength"]:
            path = "crawled/reviews_with_weaknesses.pkl"
        else:
            raise ValueError("Undefined task")

        with open(path, "rb") as f:
            raw_data = pickle.load(f)
        return raw_data

    def split_data(self, examples: List[PaperAssessFeature]) -> Dict[str, List[PaperAssessFeature]]:
        """
        Split preprocessed data into train/dev/test splits.

        :param examples: List of features

        :return: Splitted features
        """
        random.shuffle(examples)
        split1, split2 = int(len(examples) * 0.8), int(len(examples) * 0.9)
        features = {
            "train": examples[:split1],
            "dev": examples[split1:split2],
            "test": examples[split2:],
        }
        return features

    def preprocess(self) -> Tuple[PaperAssessDataset, PaperAssessDataset, PaperAssessDataset]:
        """
        Preprocess data. Read, extract features, and split features into train/dev/test.

        :return: Splitted datasets
        """
        # Load dataset
        raw_data = self.read_data()
        examples = self.read_example(raw_data)
        print("Data reading complete")

        # Split train & val & test
        features_all = self.convert_examples_to_features(examples)
        print("Tokenization complete")

        if self.task == "accepted":
            accept_true, accept_false = [], []
            for feature in features_all:
                if feature.label_ids:
                    accept_true.append(feature)
                else:
                    accept_false.append(feature)

            accept_true_split = self.split_data(accept_true)
            accept_false_split = self.split_data(accept_false)

            # integrate
            features = {}
            for i in ["train", "dev", "test"]:
                features[i] = accept_true_split[i] + accept_false_split[i]
        else:
            features = self.split_data(features_all)

        # make dataset
        train_dataset = PaperAssessDataset(features["train"])
        dev_dataset = PaperAssessDataset(features["dev"])
        test_dataset = PaperAssessDataset(features["test"])

        return train_dataset, dev_dataset, test_dataset

    def __call__(
        self,
    ) -> Tuple[
        PaperAssessDataset, PaperAssessDataset, PaperAssessDataset
    ]:  # train_dataset, dev_dataset, test_dataset
        return self.preprocess()

    def _pad_ids(self, input_ids: List[List[int]], pad_id: int) -> List[List[int]]:
        """
        Pad ids so that all ids in a batch have same length.

        :param input_ids: Batch input ids.
        :param pad_id: ID of PAD

        :return: List of padded batch ids.
        """
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
        if self.task == "accepted":
            label_ids = torch.tensor([b.label_ids for b in batch], dtype=torch.long)
        else:
            label_ids = torch.tensor(
                self._pad_ids([b.label_ids for b in batch], self.pad_id), dtype=torch.long  # type: ignore[misc]
            )

        return input_ids, label_ids

    def collate_fn_accept(
        self, batch: List[PaperAssessFeature]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(
            self._pad_ids([b.input_ids for b in batch], self.pad_id), dtype=torch.long
        )
        label_id = torch.tensor([b.label_ids for b in batch])
        return input_ids, label_id


if __name__ == "__main__":
    preprocessor = Preprocessor(
        Hyperparameter(), BertConfig.from_pretrained("bert-base-uncased"), "strength"
    )
    train, dev, test = preprocessor()
    print(len(train))
