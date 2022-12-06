import argparse
import json
from pathlib import Path
import pickle
import random
from typing import Any

import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

from hparams import Hyperparameter


def generate(
    abstract: str, model: Any, tokenizer: Any, device: torch.device, max_length: int = 15
) -> str:
    model.eval()

    abstract_ids = tokenizer.encode(abstract)[1:-1]
    abstract_ids_truncated = abstract_ids[: int(0.9 * 512)]

    gen_str = ""
    gen_ids = []
    while True:
        gen_input_str = gen_str + " [MASK]"
        gen_input_ids = tokenizer.encode(gen_input_str)[1:-1]
        input_ids_list = (
            [tokenizer.cls_token_id]
            + abstract_ids_truncated
            + [tokenizer.sep_token_id]
            + gen_input_ids
        )

        input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(device)

        with torch.no_grad():
            output_logits = model(input_ids).logits  # 1 x seq x 30522
        mask_logits = output_logits[0, -1, :]
        top_idx = torch.argmax(mask_logits).cpu().item()

        if top_idx == tokenizer.sep_token_id:
            break

        gen_ids.append(int(top_idx))
        gen_str = tokenizer.decode(gen_ids)

        if len(gen_ids) > max_length:
            break

    return gen_str


def main(args: argparse.Namespace) -> None:
    if args.task == "tldr":
        path = "crawled/reviews_with_weaknesses.pkl"
    else:
        path = "crawled/reviews_without_weaknesses.pkl"
    # dataset
    with open(path, "rb") as f:
        raw_data = pickle.load(f)

    abstract_data = []
    for paper in raw_data:
        abstract_data.append(paper["abstract"])

    abstract_data = list(set(abstract_data))

    # model
    load_dir = Path(f"checkpoints/{args.task}/bert-base-uncased")
    with open(load_dir / "hp.json") as f:
        hp_dict = json.load(f)
    hparams = Hyperparameter(**hp_dict)
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(hparams.pretrained_model)

    config = BertConfig.from_pretrained(load_dir)
    model = BertForMaskedLM.from_pretrained(load_dir, config=config).to(device)

    # test
    random.seed(512)
    idxs = random.sample(range(len(abstract_data)), args.gen_num)
    for idx in idxs:
        abstract = abstract_data[idx]

        print(f"\n[Abstract] {abstract}")
        gen_str = generate(abstract, model, tokenizer, max_length=50, device=device)
        print(f"\n[{args.task}] {gen_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=["tldr", "strength", "weakness"],
        help="Task to train",
    )
    parser.add_argument("--gpu_num", type=int, default=0, help="the number of gpu")
    parser.add_argument("--gen_num", type=int, default=4, help="the number of generation")

    args = parser.parse_args()

    print("[Start]")
    print(args)

    main(args)

    print(args)
    print("[End]")
