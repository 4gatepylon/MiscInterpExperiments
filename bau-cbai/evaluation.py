from __future__ import annotations
from typing import List, Dict, Literal
import pydantic
from dataset import Dataset
from pathlib import Path
import torch
import numpy as np
import torch.nn as nn

def model2size(model: str) -> int:
    # https://aiexpjourney.substack.com/p/the-number-of-parameters-of-gpt-4o
    if model == "gpt-4o":
        return 200*1e9
    elif model == "gpt-4o-mini":
        return 8*1e9
    elif model == "openai-community/gpt2":
        # https://huggingface.co/openai-community/gpt2
        return 124*1e6
    elif model == "openai-community/gpt2-xl":
        # https://huggingface.co/openai-community/gpt2-xl
        return 1.5*1e9
    model = model.lower()
    # Look for regex-like pattern like "-7b"
    is_ = []
    for i in range(100):
        if f"{i}b" in model:
            is_.append(i)
    if len(is_) == 1:
        return is_[0]
    raise NotImplementedError(f"Model {model} not found")
    

class ExperimentArguments(pydantic.BaseModel):
    # Our default models to evaluate, getting its own "best prompt"
    models: List[str] = [
        "gpt-4o-mini",
        "openai-community/gpt2",
        "openai-community/gpt2-xl",
        "google/gemma-2-2b",
        "google/gemma-2-9b",
    ]
    model2template_string: Dict[str, str] = {
        "gpt-4o-mini": (Path(__file__).parent / "prompt.j2").read_text(),
        "openai-community/gpt2": (Path(__file__).parent / "prompt.j2").read_text(),
        "openai-community/gpt2-xl": (Path(__file__).parent / "prompt.j2").read_text(),
        "google/gemma-2-2b": (Path(__file__).parent / "prompt.j2").read_text(),
        "google/gemma-2-9b": (Path(__file__).parent / "prompt.j2").read_text(),
    }
    # TODO(Adriano) implement this, it defines the hookpoints in order to be visualized for each model
    # model2hookpoints: Dict[str, List[str]] = {
    #     "gpt-4o-mini": [],
    #     "openai-community/gpt2": ["residual_stream"],
    #     "openai-community/gpt2-xl": ["residual_stream"],
    #     "google/gemma-2-2b": ["residual_stream"],
    #     "google/gemma-2-9b": ["residual_stream"],
    # }
    # Every model shares the same dataset though
    dataset: Dataset

class ExperimentResults(pydantic.BaseModel):
    experiment_arguments: ExperimentArguments
    # Fraction of the dataset that got correct GREEDY prediction
    model2raw_accuracy: Dict[str, float]
    # Average DISTANCE (L1) from the correct # (as predicted by linear probe)
    # (sorted in the same way as the hookpoints, so this is 1:1 with the hookpoints
    # in the `experiment_arguments.model2hookpoints`)
    model2probes_accuracy: Dict[str, List[float]]
    # The functional that the linear probe learned for each hookpoint (also sorted
    # like `model2probes_accuracy, so all these are 1:1:1`. The functional is a row
    # vector corresponding to a direction in vector space. It has one last element
    # corresponding to a bias term.
    # 
    # You will want to convert to torch or numpy arrays.
    model2probes_functionals: Dict[str, List[List[float]]]
    def get_model2probes_functionals_torch(self, format: Literal["torch", "numpy", "nn.Linear"] = "torch", device: str = "cpu") -> Dict[str, torch.Tensor]:
        assert format in ["torch", "numpy", "nn.Linear"]
        model2_functionals = {
            # Don't do a grid because these might be different lengths (i.e. different hookpoints)
            model: [torch.Tensor(value[i]).to(device if format != "numpy" else "cpu") for i in range(len(value))]
            for model, value in self.model2probes_functionals.items()
        }
        if format == "numpy":
            model2_functionals = {
                model: [v.numpy() for v in v_list] for model, v_list in model2_functionals.items()
            }
        elif format == "nn.Linear":
            # Must have one linear probe per hookpoint
            assert all(all(v.ndim == 1 for v in v_list) for v_list in model2_functionals.values())
            model2nn_linears = {
                model: [nn.Linear(v.shape[0] - 1, 1, bias=True) for v in v_list]
                for model, v_list in model2_functionals.items()
            }
            for model in model2nn_linears:
                for i in range(len(model2nn_linears[model])):
                    model2nn_linears[model][i].weight.data = model2_functionals[model][i][:-1]
                    model2nn_linears[model][i].bias.data = model2_functionals[model][i][-1]
                    model2nn_linears[model][i].to(device) # redundant, but a bit paranoid
                    # This can help efficiency or whatever a bit
                    model2nn_linears[model][i].eval()
                    model2nn_linears[model][i].requires_grad_(False)
            # Rename for the return below
            model2_functionals = model2nn_linears
        return model2_functionals

    def plot_scaling_violin(self):
        """
        Plot a violin plot with the models SORTED by their size from left to right.

        It has two components: the violin (which is the distribution of the accuracies from the linear probes)
        and the raw accuracy (which is a bright red dot somewhere on that x-axis center-line).
        """
        raise NotImplementedError("Not implemented")
