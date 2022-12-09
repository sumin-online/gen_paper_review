import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig, BertForMaskedLM
import wandb

from hparams import Hyperparameter
from preprocess import Preprocessor


def get_acc(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    predictions = (scores > 0.5).long()
    num_correct = (predictions == y).sum()
    num_samples = predictions.size(0)

    acc = num_correct / num_samples
    return acc


def validate(model: Any, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Tuple[float, float]:  # type: ignore[type-arg]
    model.eval()
    total_loss = 0
    num_correct = 0
    num_samples = 0

    for batch in loader:
        input_ids, label_ids = batch
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            output = model(input_ids)
            prediction = output.logits[:, 0, :].squeeze()
            if len(prediction.shape) < 1:  # for batch size = 1
                prediction = prediction.reshape(1)

            scores = torch.sigmoid(prediction)
            loss = criterion(scores.float(), label_ids.float())

            predictions = (scores > 0.5).long()
            num_correct += (predictions == label_ids).sum()
            num_samples += predictions.size(0)

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    avg_acc = (num_correct / num_samples).item()  # type: ignore[attr-defined]
    return avg_loss, avg_acc


def test(
    model_save_path: Union[str, Path],
    model: nn.Module,
    test_loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    print("[Testing]")
    dic = torch.load(model_save_path)
    model.load_state_dict(dic["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, device, criterion)
    test_dict = {"test_loss": test_loss, "test_acc": test_acc}
    return test_dict


def train(args: argparse.Namespace) -> None:
    if args.use_wandb:
        wandb.login(key="")
        run = wandb.init(
            name="[New Data] Paper Accpetance",  # Wandb creates random run names if you skip this field
            reinit=True,  # Allows reinitalizing runs when you re-run this cell
            # run_id = # Insert specific run id here if you want to resume a previous run
            # resume = "must" # You need this to resume previous runs, but comment out reinit = True when using this
            project="IDL",  # Project should be created in your wandb account
            # config = configParaser # Wandb Config for your run
            entity="gyuseoklee",
        )
    else:
        run = None

    # Hyperparameters
    hp = Hyperparameter()
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    save_dir = Path(hp.save_dir) / args.task / hp.pretrained_model

    # Preprocessor
    config = BertConfig.from_pretrained(hp.pretrained_model)
    preprocessor = Preprocessor(hp, config=config, task=args.task)

    # train/dev/test dataset
    train_dataset, dev_dataset, test_dataset = preprocessor()

    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=preprocessor.collate_fn_accept,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=preprocessor.collate_fn_accept,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=preprocessor.collate_fn_accept,
    )

    # Model
    model = BertForMaskedLM.from_pretrained(hp.pretrained_model, config=config).to(device)
    model.cls.predictions.decoder = nn.Linear(768, 1).to(device)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=hp.learning_rate, eps=1e-8)  # type: ignore[arg-type]
    criterion = nn.BCELoss().to(device)

    if args.test_only:
        test_dict = test(args.model_pretrain_path, model, test_loader, device, criterion)
        print(test_dict)
        if args.use_wandb:
            wandb.log(test_dict)
        return

    min_dev_loss = float("inf")
    global_step = 0

    for epoch in range(hp.max_epochs):  # hp.max_epochs):
        print(f"[Epoch : {epoch}]")
        model.train()
        num_correct = 0
        num_samples = 0
        total_loss = 0

        print("[Training]")
        for idx, batch in enumerate(train_loader):
            input_ids, label_ids = batch
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)

            output = model(input_ids)
            prediction = output.logits[:, 0, :].squeeze()
            if len(prediction.shape) < 1:  # for batch size = 1
                prediction = prediction.reshape(1)

            scores = torch.sigmoid(prediction)
            loss = criterion(scores.float(), label_ids.float())
            total_loss += loss.item()

            predictions = (scores > 0.5).long()
            num_correct += (predictions == label_ids).sum()
            num_samples += predictions.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            train_step_loss = total_loss / (idx + 1)
            train_step_acc = (num_correct / num_samples).item()  # type: ignore[attr-defined]

            # validate
            dev_loss, dev_acc = None, None
            if global_step % 20 == 0:
                dev_loss, dev_acc = validate(model, dev_loader, device, criterion)

                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss

                    # Save model
                    model.save_pretrained(save_dir)
                    config.save_pretrained(save_dir)
                    with open(save_dir / "hp.json", "w") as f:
                        json.dump(hp.dict(), f, indent=4, ensure_ascii=False)

                    status = {"global_step": global_step, "epoch": epoch}
                    with open(save_dir / "status.json", "w") as f:
                        json.dump(status, f, indent=4, ensure_ascii=False)

                    # Notice model saved
                    model_save_path = args.model_save_path
                    torch.save({"model_state_dict": model.state_dict()}, model_save_path)

                    print(f"Model saved at step {global_step} / epoch {epoch}")

            # result

            step_result = {
                "train_step_acc": train_step_acc,
                "train_step_loss": train_step_loss,
                "valid_acc": dev_acc,
                "valid_loss": dev_loss,
            }

            print(
                f"[Epoch : {epoch:04d}] global_step : {global_step} | loss : {dev_loss:.6f} | acc: {dev_acc:.6f}"
            )
            print(step_result)
            if args.use_wandb:
                wandb.log(step_result)

        # epoch_log
        if args.use_wandb:
            epoch_result = {"train_epoch_loss": train_step_loss, "train_epoch_acc": train_step_acc}
            wandb.log(epoch_result)

    # Test
    test_dict = test(model_save_path, model, test_loader, device, criterion)
    print(test_dict)
    if args.use_wandb:
        wandb.log(test_dict)
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=["tldr", "strength", "weakness", "accepted"],
        help="Task to train",
        default="accepted",
    )
    parser.add_argument("--gpu_num", type=int, default=0, help="the number of gpu")
    parser.add_argument(
        "--model_pretrain_path", type=str, default="./checkpoints/accepted/accepted.pth"
    )
    parser.add_argument(
        "--model_save_path", type=str, default="./checkpoints/accepted/accepted.pth"
    )
    parser.add_argument(
        "--test_only", type=bool, default=False, help="Whether only test mode or not"
    )
    parser.add_argument("--run_wandb", action="store_true", help="Use WandB logging")

    args = parser.parse_args()

    print("[Starting]")
    print(args)

    train(args)

    print(args)
    print("[Ending]")
