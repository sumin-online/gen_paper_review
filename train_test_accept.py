import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig, BertForMaskedLM

from hparams import Hyperparameter
from preprocess import Preprocessor

import torch
import torch.nn as nn
import wandb

def get_acc(scores, y):
    predictions = (scores > 0.5).long()
    num_correct += (predictions == y).sum()
    num_samples += predictions.size(0)
    
    acc = num_correct/num_samples
    return acc


def validate(model: Any, loader: DataLoader, device: torch.device, Sigmoid, criterion):  # type: ignore[type-arg]
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
            prediction = output.logits[:,-1,:].squeeze()
            if len(prediction.shape) < 1: # for batch size = 1
                prediction = prediction.reshape(1)
            
            scores = Sigmoid(prediction)
            loss = criterion(scores.float(), label_ids.float())

            predictions = (scores > 0.5).long()
            num_correct += (predictions == label_ids).sum()
            num_samples += predictions.size(0)

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    avg_acc  = (num_correct / num_samples).item()
    return avg_loss, avg_acc


def train(args: argparse.Namespace) -> None:
    wandb.login(key="e0408f5d7b96be3d00be30b39eda0f1e259672ed")
    run = wandb.init(
        name = "Paper Accpetance", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "IDL", ### Project should be created in your wandb account 
        #config = configParaser ### Wandb Config for your run
        entity = "gyuseoklee"
    )

    # Hyperparameters
    hp = Hyperparameter()
    device = torch.device(hp.device)
    save_dir = Path(hp.save_dir) / hp.pretrained_model

    # Preprocessor
    config = BertConfig.from_pretrained(hp.pretrained_model)
    preprocessor = Preprocessor(hp, config=config)

    # train/dev/test dataset
    train_dataset, dev_dataset, test_dataset = preprocessor(args.task)

    train_loader = DataLoader(
        train_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=preprocessor.collate_fn_accept
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=preprocessor.collate_fn_accept
    )
    test_loader = DataLoader(
        test_dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=preprocessor.collate_fn_accept
    )

    # Model
    model = BertForMaskedLM.from_pretrained(hp.pretrained_model, config=config).to(device)
    model.cls.predictions.decoder = nn.Linear(768,1).to(device)

    # check
    # x,y = next(iter(train_loader))
    # output = model(x.to(device))
    # print("output", output.logits.shape) # 8 x 512 x 1

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
    Sigmoid = nn.Sigmoid().to(device)


    min_dev_loss = float("inf")
    global_step = 0
    patience = 0
    early_stopping = False


    for epoch in range(10):#hp.max_epochs):
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
            prediction = output.logits[:,-1,:].squeeze()
            if len(prediction.shape) < 1: # for batch size = 1
                prediction = prediction.reshape(1)
            
            scores = Sigmoid(prediction)
            loss = criterion(scores.float(), label_ids.float())
            total_loss += loss.item()

            predictions = (scores > 0.5).long()
            num_correct += (predictions == label_ids).sum()
            num_samples += predictions.size(0)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            train_step_loss = total_loss / (idx+1)
            train_step_acc = (num_correct / num_samples).item()
            
            # validate
            dev_loss, dev_acc = None, None
            if global_step % 1 == 0:
                #print("[Validating]")
                dev_loss, dev_acc = validate(model, dev_loader, device, Sigmoid, criterion)
                #print(f"[Epoch:{epoch}] Step {global_step} | Dev loss: {dev_loss:.6f} Dev acc: {dev_acc:.6f}")

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
                    model_save_path = "./checkpoints/accepted.pth"
                    torch.save({"model_state_dict" : model.state_dict()}, model_save_path)

                    print(f"Model saved at step {global_step} / epoch {epoch}")
                    patience = 0

                # else:
                #     patience += hp.validation_steps
                #     if patience > hp.patience_steps:
                #         print(f"Early stopping: No improvement in {patience} epochs")
                #         early_stopping = True
                        #break
            # result
            step_result = {"train_step_acc" : train_step_acc,
                           "train_step_loss": train_step_loss,
                           "valid_acc" : dev_acc,
                           "valid_loss": dev_loss}

            print(f"[Epoch : {epoch:04d}] global_step : {global_step}")
            print(step_result)
            wandb.log(step_result)

        # epoch_log
        epoch_result = {"train_epoch_loss": train_step_loss,
                        "train_epoch_acc": train_step_acc}
        wandb.log(epoch_result)

        # if early_stopping:
        #     break

    # Test
    print("[Testing]")

    dic = torch.load(model_save_path)
    model.load_state_dict(dic["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, device, Sigmoid, criterion)
    test_dict = {"test_loss": test_loss, "test_acc":test_acc}
    
    print(test_dict)
    wandb.log(test_dict)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=["tldr", "strength", "weakness", "accepted"],
        help="Task to train",
        default = "accepted" 
    )
    args = parser.parse_args()

    print("[Starting]")
    print(args)

    train(args)

    print(args)
    print("[Ending]")

