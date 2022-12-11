import json
from pathlib import Path
from typing import Union

import streamlit as st
import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

from generate import generate
from hparams import Hyperparameter


def get_analysis(abstract: str, task: str) -> Union[str, float]:
    """
    Run analysis generation/prediction for single abstract-single task.

    :param abstract: Abstract text
    :param task: Task to run (tldr, strength, weakness, accepted)

    :return: Acceptance score or generated analysis
    """
    torch.cuda.empty_cache()
    load_dir = Path(f"checkpoints/{task}/bert-base-uncased")
    with open(load_dir / "hp.json") as f:
        hp_dict = json.load(f)
    hparams = Hyperparameter(**hp_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(hparams.pretrained_model)

    config = BertConfig.from_pretrained(load_dir)
    if task == "accepted":
        model = BertForMaskedLM.from_pretrained("bert-base-uncased", config=config).to(device)
        model.cls.predictions.decoder = nn.Linear(768, 1).to(device)

        model.load_state_dict(torch.load("checkpoints/accepted/accepted.pth")["model_state_dict"])

        model.eval()

        inputs = tokenizer(abstract, truncation=True)
        inputs = {key: torch.tensor([val]).to(device) for key, val in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        prediction = output.logits[:, 0, :].squeeze()
        if len(prediction.shape) < 1:  # for batch size = 1
            prediction = prediction.reshape(1)

        scores = torch.sigmoid(prediction).item()
        return scores

    else:
        model = BertForMaskedLM.from_pretrained(load_dir, config=config).to(device)
        gen_str = generate(abstract, model, tokenizer, max_length=50, device=device)

        return gen_str


INITIAL_STATE = {"abstract": "", "tldr": "", "strength": "", "weakness": "", "accepted": 0.0}


def main() -> None:
    """Run streamlit app."""
    for key, val in INITIAL_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = val

    st.title("IDL Project Demo")
    st.header("Paper Review Generator")

    abstract = st.text_area(label="Insert Abstract")

    button_columns = st.columns(4)
    with button_columns[0]:
        gen_tldr = st.button("TL;DR")

    with button_columns[1]:
        gen_strength = st.button("Paper Strength")

    with button_columns[2]:
        gen_weakness = st.button("Paper Weakness")

    with button_columns[3]:
        run_accepted = st.button("Accepted?")

    with st.spinner("Analyzing paper..."):
        if gen_tldr:
            tldr = get_analysis(abstract, "tldr")
            st.session_state["tldr"] = tldr

        if gen_strength:
            strength = get_analysis(abstract, "strength")
            st.session_state["strength"] = strength

        if gen_weakness:
            weakness = get_analysis(abstract, "weakness")
            st.session_state["weakness"] = weakness

        if run_accepted:
            accepted_score = get_analysis(abstract, "accepted")
            st.session_state["accepted"] = accepted_score

    st.write("* * *")
    st.header("Result")

    st.text_area(label="TL;DR", value=st.session_state["tldr"], disabled=True)

    st.text_area(label="Strength", value=st.session_state["strength"], disabled=True)

    st.text_area(label="Weakness", value=st.session_state["weakness"], disabled=True)

    st.text("Acceptance score")
    st.text(st.session_state["accepted"])
    st.write("**Accepted!**" if st.session_state["accepted"] > 0.5 else "Rejected")


if __name__ == "__main__":
    main()
