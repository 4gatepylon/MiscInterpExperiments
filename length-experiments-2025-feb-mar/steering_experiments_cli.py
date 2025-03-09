from __future__ import annotations
"""
This is a CLI version of `steering_experiments.py` with a focus on ONLY running the inference necessary to generate the steering outputs dataset.

It can be fairly slow, so we want to make sure to run this shit on multiple devices. Currently we are running:
magnitudes: List[float] = [1.0, 2.0, 4.0, 8.0, 20.0]
layer_nums: List[int] = [2, 8, 12, 16]
and we have all 1.0 and 2.0 and two 4.0's so we are going to run 4.0 and higher on different devices...

NOTE: almost all this code is copied from `steering_expeirments.ipynb`.

Open a json file on a path like:
```
{
    "magnitudes": [3.0, 4.0, 10.0, 20.0],
    "layer_nums": [3, 4, 5, 6, 14],
    "device": "cuda:1"
}
```
and on that device run on the product of all those above.
"""
from datetime import datetime
from pathlib import Path
import json
from contextlib import contextmanager
import contextlib
import uuid
import io
from copy import deepcopy
from torch.utils.hooks import RemovableHandle
from jinja2 import Template
import pydantic
import itertools
import os
import tqdm
import matplotlib.pyplot as plt
import time
import math
from datasets import concatenate_datasets
from collections import defaultdict
import gc
from safetensors.torch import save_file as safetensors_save_file
from safetensors.torch import load_file as safetensors_load_file
import random
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import numpy as np
# import zipnn # TODO(Adriano) start using ZNN to save activations for better performance on storage: https://github.com/zipnn/zipnn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dotenv
from jaxtyping import Float, Int
import einops
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import tempfile
import click

system_prompt = """You are a helpful and informative assistant that can answer questions about the world. Try
to cause no harm and aim primarily to give information to users who may be curious about different facts."""
class EncodingArgs(pydantic.BaseModel):
    test_size_pct: float = 0.2
    llm_tok_max_length: int = 4096
    llm_tok_max_prompt_length: Optional[int] = None
    system_prompt: Optional[str] = system_prompt
    message_key: Optional[str | Dict[str, str]] = "prompt"

class DatasetEncoder:
    """
    A static-like (but stateful) class to coordinate the creation of a TON of tokenized and standardized
    dataset objects that we can use to basically do batched inference. This should make it really easy to
    run inference on huggingface datasets with various LLMs.

    The main way to do this is to:
    1. Init
    2. Call `get_trainable_datasets` to get something that you can train/run inference on.
    """
    def __init__(
            self,
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            datasets: Dict[str, Dataset],
            device: Optional[str]=None,
            encoding_args: EncodingArgs=EncodingArgs()
        ) -> None:
        """models, datasets are { model/dataset_name: model/dataset }"""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.raw_datasets: Dict[str, Dataset] = datasets
        self.train_test_raw_datasets: Dict[str, Tuple[Dataset, Dataset]] = {}
        self.train_test_convo_datasets: Dict[str, Tuple[Dataset, Dataset]] = {}
        self.train_test_tok_datasets: Dict[str, Tuple[Dataset, Dataset]] = {}
        self.encoding_args = encoding_args

        # Create some variables to make this faster to query
        self.llm_tok_max_length = self.encoding_args.llm_tok_max_length
        self.llm_tok_max_prompt_length = self.encoding_args.llm_tok_max_prompt_length if self.encoding_args.llm_tok_max_prompt_length is not None else self.llm_tok_max_length // 4 # fmt: skip
        self.test_size_pct = self.encoding_args.test_size_pct
        # Create a default dictionary for the message key (note this is meant to tell you the message key per dataset;
        # it's not perfect since you might have multiple messages you want to format, but we'll handle that in a future
        # version)
        self.default_message = self.encoding_args.message_key if isinstance(self.encoding_args.message_key, str) else self.encoding_args.message_key.get("default", None) # fmt: skip
        self.message_key_dict = {} if (self.encoding_args.message_key is None or isinstance(self.encoding_args.message_key, str)) else self.encoding_args.message_key # fmt: skip
        self.message_key_dict = defaultdict(lambda: self.default_message, self.message_key_dict) # fmt: skip
        # ....
        self.system_prompt = self.encoding_args.system_prompt

        # Used instead of allowing user template, since Llama sort of forces there to be a date string
        # and if it changes every time we run this, it will be more non-deterministic and that could
        # be problematic for our results
        self.date_string: str = "23 Jan 2025" 

        # Does more logging/printign, etc...
        self.debug_mode: bool = True

    def get_train_test_raw_datasets(self) -> None:
        """split train/test and filter out prompts that are too long"""
        for dataset_name, dataset in self.raw_datasets.items():
            message_key = self.message_key_dict[dataset_name]
            assert message_key is not None, f"message_key_dict={self.message_key_dict}, dataset_name={dataset_name}"
            assert dataset is not None
            dataset = dataset.shuffle(seed=42)
            dataset = dataset.filter(lambda x: len(x[message_key]) <= self.llm_tok_max_length)
            assert dataset is not None
            dataset = dataset.filter(lambda x: len(x[message_key]) <= self.llm_tok_max_prompt_length)
            assert dataset is not None
            test_size = int(len(dataset) * self.test_size_pct)
            train_size = len(dataset) - test_size
            assert test_size > 0 and train_size > 0
            train_dataset = dataset.select(range(train_size))
            test_dataset = dataset.select(range(train_size, len(dataset)))
            assert len(train_dataset) + len(test_dataset) == len(dataset)
            assert len(train_dataset) == train_size
            assert len(test_dataset) == test_size
            self.train_test_raw_datasets[dataset_name] = (train_dataset, test_dataset)
    
    def format_convo(self, x: dict, response_key: Optional[str]=None, dataset_name: Optional[str]=None) -> dict:  # fmt: skip
        message_key = self.message_key_dict[dataset_name] if dataset_name is not None else self.default_message # fmt: skip
        assert message_key is not None, f"dataset_name={dataset_name}, message_key_dict={self.message_key_dict}"
        if response_key is not None:
            raise NotImplementedError("We do not use responses yet... because we want native model generations")
        convo = {'conversation': []}
        if self.system_prompt is not None:
            convo['conversation'].append({'role': 'system', 'content': self.system_prompt})
        if message_key is not None:
            convo['conversation'].append({'role': 'user', 'content': x[message_key]})
        x.update(convo)
        return x
    
    def tok_and_tmpl(self, x: dict, max_length: Optional[int] = None) -> dict: # fmt: skip
        shared_kwargs = {"date_string": self.date_string, "truncation": False } # for unk. reasons adding gen. prompt is bad; fmt: skip
        # NOTE: for unknown reasons, left padding SUCKS and default is right padding, so we just use right padding LOL
        if max_length is not None:
            shared_kwargs["max_length"] = max_length
            shared_kwargs["padding"] = "max_length"
        x.update({
            # https://huggingface.co/docs/transformers/v4.35.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.apply_chat_template.add_generation_prompt; fmt: skip
            # NOTE: we do not transition to device becasue it doesn't actually do it
            'input_ids': self.tokenizer.apply_chat_template(x["conversation"], tokenize=True, return_tensors="pt", **shared_kwargs), # fmt: skip
            'input_templated': self.tokenizer.apply_chat_template(x["conversation"], tokenize=False, **shared_kwargs), # fmt: skip
        })
        return x
    
    def get_trainable_datasets(self) -> Dict[str, Tuple[Dataset, Dataset]]:
        """
        This is basically the main method for this function. NOTE that the output will be in a specific format:
        - Dataset object of dict-like objects
        - Each dict-like object has the following keys:
            - 'input_ids': a tensor of token ids (what you put into your model)
            - 'input_templated': a string of the templated input (so you can sanity check the tokenization)
            - 'conversation': a list of dicts with keys 'role' and 'content' (OpenAI API-like format)
            - ... possibly whatever else was there before
        """
        # 1. Get raw datasets
        if self.debug_mode:
            print("Getting train/test raw datasets...")
        self.get_train_test_raw_datasets()
        assert len(self.train_test_raw_datasets) == len(self.raw_datasets)
        
        # 2. Format (or more technically just like a step to do step 3 ngl.)
        if self.debug_mode:
            print("Formatting datasets to be in conversation format...")
        pbar = self.train_test_raw_datasets.items()
        if self.debug_mode:
            pbar = tqdm.tqdm(pbar, total=len(self.train_test_raw_datasets), desc="Formatting datasets to be in conversation format") # fmt: skip
        for dataset_name, (train_raw, test_raw) in pbar:
            self.train_test_convo_datasets[dataset_name] = (train_raw.map(lambda x: self.format_convo(x, dataset_name=dataset_name)), test_raw.map(lambda x: self.format_convo(x, dataset_name=dataset_name))) # fmt: skip
        assert len(self.train_test_convo_datasets) == len(self.train_test_raw_datasets)
        
        # 3. Tokenize
        if self.debug_mode:
            print("Applying chat template and tokenizing datasets...")
            pbar = self.train_test_convo_datasets.items()
            if self.debug_mode:
                pbar = tqdm.tqdm(pbar, total=len(self.train_test_convo_datasets), desc="Applying chat template and tokenizing datasets...") # fmt: skip
            # 3.1. Conversion
            for dataset_name, (train_convo, test_convo) in pbar:
                # 3.1.1. Tokenize to get max_length
                _train_tok, _test_tok = train_convo.map(self.tok_and_tmpl), test_convo.map(self.tok_and_tmpl) # fmt: skip
                # 3.1.2. Sanity check lengths and get those lengths
                lengths_train = np.array([len(x['input_ids'][0]) for x in _train_tok])
                lengths_test = np.array([len(x['input_ids'][0]) for x in _test_tok])
                assert np.max(lengths_train).item() <= self.encoding_args.llm_tok_max_length
                assert np.max(lengths_test).item() <= self.encoding_args.llm_tok_max_length
                if self.debug_mode:
                    max_length_train, min_length_train = np.max(lengths_train).item(), np.min(lengths_train).item()
                    max_length_test, min_length_test = np.max(lengths_test).item(), np.min(lengths_test).item()
                    print(f"max length train: {max_length_train}, min length train: {min_length_train}") # DEBUG
                    print(f"max length test: {max_length_test}, min length test: {min_length_test}") # DEBUG
                # 3.1.3. Create with padding (turns out for Llama model this is right padding, but in general what
                #   matters is that we can generate and that we get OK/good looking responses relative to the capabilities
                #   of the model)
                train_tok = _train_tok.map(lambda x: self.tok_and_tmpl(x, max_length=max_length_train))
                test_tok = _test_tok.map(lambda x: self.tok_and_tmpl(x, max_length=max_length_test))
                self.train_test_tok_datasets[dataset_name] = (train_tok, test_tok) # fmt: skip
            # 3.2 Sanity check
            assert len(self.train_test_tok_datasets) == len(self.train_test_convo_datasets)
        return self.train_test_tok_datasets

class DecodingArgs(pydantic.BaseModel):
    """
    Args for the decoder.
    """
    # Turns out on my machine 511 is really good and 513 is really bad (probably some cache thing)
    batch_size: int | Tuple[int, int] = 511
    generation_kwargs: Dict[str, Any] = {
        # NOTE: we keep these as a dict to feed them into the **kwargs for a huggingface (generate) function
        "do_sample": True,
        "temperature": 1,
        "top_k": 50,
        "top_p": 0.95,
        # This may seem dumb, but basically, we actually don't do generation itself, per-say,
        # all the time, i.e. we only want to get the activations from the model
        "max_new_tokens": 1
    }

class BatchEncodedOutput(pydantic.BaseModel):
    """
    A class to store the output of a batch of encoded outputs.
    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    generations: List[str] # Includes the generations :P
    batch_size_optimized: Optional[int] = None
    input_ids_pt: Optional[torch.Tensor] = None
    activations: Optional[Dict[str, torch.Tensor]] = None

class DatasetDecoder:
    """
    A static-like (but also stateful, like above) class that lets you:
    1. Run a generation loop
        - Normally
        - With hooks (and the ability to apply the hook on a single token or multiple SPECIFIC tokens in
            the generation of the .generate() call)
    2. Decode the generations in strings so we can print them.
    3. Get the activations from the model
        and store them on disk efficiently-ish....
    """
    @staticmethod
    def batch_decode(
        # self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        questions: Dataset | torch.Tensor,
        # TODO(Adrianoh) transition to the pydantic schema :P
        # NOTE: that after some optimization 511 was found to be good for performance, and
        # we suspect it might have something to do with some kind of cache size
        batch_size: int | Tuple[int, int] = 511,
        generation_kwargs: Dict[str, Any] = {
            "do_sample": True,
            "temperature": 1,
            "top_k": 50,
            "top_p": 0.95,
            # Notice how these default settings mean that we roughly are in line with the amount of allowable
            # generation as defined before
            "max_new_tokens": 1 #1024
        },
        device='cuda:0',
        optimize_batch_size: bool = False,
        return_input_ids_pt: bool = False,
        return_optimal_batch_size: bool = False,
        return_activations: bool = False, # TODO(Adrianoh) modify to make it easier to store the activations in a sort of "streaming" way i.e. don't OOM lol
        debug: bool = False,
    ) -> BatchEncodedOutput:
        """
        Run a bunch of inference and get the input+output+activations (if you so wish).
        Returns:
            - The input dataset in the tensor form for reuse if requested
            - The input dataset in the tokenized form for display purposes
            - The activations from the model if requested
        """
        batch_size_lower, generations, activations = None, None, None
        try:
            if isinstance(questions, Dataset):
                all_input_ids = []
                for input_ids in tqdm.tqdm(questions['input_ids'], total=len(questions), desc="Converting input_ids to tensor (and moving to device)"): # fmt: skip
                    assert len(input_ids) == 1, f"input_ids={input_ids}, len(input_ids)={len(input_ids)}"
                    all_input_ids.append(input_ids[0])
                assert len(all_input_ids) == len(questions['input_ids']), f"len(all_input_ids)={len(all_input_ids)} != len(questions['input_ids'])={len(questions['input_ids'])}" # fmt: skip
                try:
                    questions = torch.Tensor(all_input_ids).long().to(device) # transition all to device in one big chunk
                except Exception as e:
                    print("DISTRIBUTION OF LENGTHS")
                    lengths = np.array([len(x) for x in all_input_ids])
                    print(f"min={np.min(lengths)}, max={np.max(lengths)}, mean={np.mean(lengths)}, std={np.std(lengths)}") # fmt: skip
                    raise e
            else:
                assert isinstance(questions, torch.Tensor), f"questions is not a torch.Tensor but a {type(questions)}" # fmt: skip
                questions = questions.to(device)
            assert isinstance(questions, torch.Tensor), f"questions is not a torch.Tensor but a {type(questions)}" # fmt: skip
            if debug:      
                print("Shape of questions: {}".format(questions.shape))
            if optimize_batch_size:
                # TODO(Adrianoh) I will do this one day; it will speed up my workflow when I need to switch GPUs n shit
                raise NotImplementedError
            elif isinstance(batch_size, tuple):
                raise ValueError(f"optimize_batch_size is False but batch_size is an int, make sure it's a tuple range of batch sizes!, batch_size={batch_size}, type={type(batch_size)}") # fmt: skip
            assert isinstance(batch_size, int)
            batch_size_lower = batch_size

            # ...
            generations: List[str] = []
            activations: List[torch.Tensor] = []
            times_per_batch: np.ndarray = np.zeros(math.ceil(len(questions) / batch_size_lower))
            for iter_i, i in enumerate(tqdm.trange(0, len(questions), batch_size_lower)):
                time_start = time.time()
                # Collect the batch
                left_idx, right_idx = i, min(i+batch_size, len(questions))
                toks_pt = questions[left_idx:right_idx]
                assert isinstance(toks_pt, torch.Tensor)
                assert toks_pt.shape[0] <= batch_size_lower
                assert toks_pt.device == model.device
                generation_tensor = model.generate(
                    input_ids=toks_pt,
                    **generation_kwargs
                )
                if return_activations:
                    # print('getting acts out yay') # debug
                    # TODO(Adrianoh) not only inputs plz wtf?
                    _acts_out = model(toks_pt, output_hidden_states=True) # only inputs yolooooo
                    _acts = _acts_out.hidden_states
                    # print("\n" + "\n".join(f"{x.shape}" for x in _acts))
                    shape0 = _acts[0].shape
                    assert all(x.shape == shape0 for x in _acts)
                    _stacked_acts = torch.stack(_acts, dim=0)
                    activations.append(_stacked_acts.detach().cpu()) # TODO(Adrianoh) optimize batched transfers to CPU so we run faster
                assert generation_tensor.shape[0] <= batch_size_lower # batch = 1
                assert len(generation_tensor.shape) == 2 # batch seq
                _gens = generation_tensor.to("cpu").detach()
                
                these_generations = tokenizer.batch_decode(_gens, skip_special_tokens=True)
                # for generation in generations:
                #     print(generation)
                #     print("="*100)
                generations.extend(these_generations)
                time_end = time.time()
                times_per_batch[iter_i] = time_end - time_start
                # This should be fast since it's a small number of averages
                if debug:
                    _running_avg = np.mean(times_per_batch[:iter_i+1])
                    _running_std = np.std(times_per_batch[:iter_i+1])
                    _running_max = np.max(times_per_batch[:iter_i+1])
                    _running_min = np.min(times_per_batch[:iter_i+1])
                    print(f"Secs per batch: {times_per_batch[iter_i]:.4f}; running avg={_running_avg:.4f}; std={_running_std:.4f}; max={_running_max:.4f}; min={_running_min:.4f}") # fmt: skip
            # print("="*100)
            # print("ACTIVATIONS SHAPES")
            # print("\n" + "\n".join(f"{x.shape}" for x in activations)) # Debug
            # print("="*100)
            assert all(len(act.shape) == 4 for act in activations)
            activations = torch.cat(activations, dim=1) if len(activations) > 0 else None # NOTE: this may not work because 
            hidden_dim_size = activations.shape[-1] if activations is not None else None
            assert questions is None or len(questions.shape) == 2
            max_input_length = questions.shape[1] if questions is not None else None # batch x seq
            expected_activations_shape = (len(model.model.layers)+1, len(generations), max_input_length, hidden_dim_size) if activations is not None else None # fmt: skip
            assert activations is None or (activations.shape == expected_activations_shape), f"Activations shape is {activations.shape} but expected {expected_activations_shape}"
        finally:
            try:
                del _acts_out, _acts
                gc.collect()
                torch.cuda.empty_cache()
            except NameError:
                pass
        assert batch_size_lower is not None and generations is not None, f"Your program probably failed :>"
        return BatchEncodedOutput(
            batch_size_optimized=batch_size_lower if return_optimal_batch_size else None,
            input_ids_pt=questions if return_input_ids_pt else None,
            generations=generations,
            activations=activations if return_activations else None
        )

def add_zero_bias_to_all_layers_down_proj_bias(model: LlamaForCausalLM) -> None:
    random_linear = torch.nn.Linear(10, 10, bias=True)
    assert isinstance(random_linear.bias, torch.nn.Parameter), f"random_linear.bias must be a Parameter, but is {type(random_linear.bias)}" # fmt: skip
    random_linear.bias.data = torch.zeros_like(random_linear.bias.data)
    del random_linear
    print(model.model.layers[0].mlp.down_proj.weight.shape)
    # raise NotImplementedError # DEBUG
    hidden_dim = model.model.layers[0].mlp.down_proj.weight.shape[0]
    for layer in model.model.layers:
        assert isinstance(layer.mlp.down_proj, torch.nn.Linear), f"layer.mlp.down_proj must be a Linear, but is {type(layer.mlp.down_proj)}" # fmt: skip
        layer.mlp.down_proj.bias = torch.nn.Parameter(data=torch.zeros(hidden_dim, device=model.device))
def insert_steering_vector_into_model_bias(
        model: LlamaForCausalLM,
        layer_nums: List[int],
        steering_vecs: torch.Tensor,
        scale: float = 1.0,
        relative_token_idx: int = -1, # -1 => last token, -2 => 2nd to last token, etc... (i.e. the front could have been clipped off); fmt: skip\
        allow_failure: bool = False
) -> bool:
    """
    Insert the steering vector into the model bias at the specified layer layer_nums. Unfortunately, it is not supported
    to insert in layer_nums 0 (embedding layer) or layer_nums 1 (first layer) because there is no good way to ADD it into a bias
    (i.e. it is not supported to do model.model.layers[0].hook_resid_post.bias.data += steering_vec).

    NOTE layer_nums n means to modify layers array layer_nums n-2 (so index 2 is input to layer 2 and acts on layer index 0: one subtract
    from inserting in the bias of the previous down_proj, and one subtract from having 0 => embedding layer).

    Examples:
        - Want to change layer 2: use 2 and it will change array element 0 at the bias right in the end
        - Want to change layer 3: use 3 and it will change array element 1 at the bias right in the end
        - Want to change layer 4: use 4 and it will change array element 2 at the bias right in the end
        ...
        - Want to change the last layer? use len(model.model.layers)
    
    NOTE: this will modify the model in place and it WILL cause the model to apply the steering vector at EVERY SINGLE TOKEN.
    NOTE: to REMOVE the steering vector just set scale=-1.0.
    """
    model_hidden_size = model.model.embed_tokens.weight.data.shape[1]
    if 0 in layer_nums or 1 in layer_nums:
        # TODO(Adriano) I think this can be done by just adding into the embeddings
        raise NotImplementedError("Cannot insert steering vector into bias at index 0 or 1 because it is not supported") # fmt: skip
    if max(layer_nums) > len(model.model.layers):
        raise ValueError(f"Cannot insert steering vector into bias at layer_num={max(layer_nums)} because it is greater than the number of layers={len(model.model.layers)}") # fmt: skip
    if len(set(layer_nums)) != len(layer_nums):
        raise ValueError(f"layer_nums={layer_nums} must be unique") # fmt: skip
    assert steering_vecs.shape[0] == len(model.model.layers)+1, f"Steering vectors shape={steering_vecs.shape}"
    assert steering_vecs.shape[-1] == model_hidden_size, f"Steering vectors shape={steering_vecs.shape}"
    assert all(layer.mlp is not None for layer in model.model.layers), f"All layers must have a MLP; model.model.layers={model.model.layers}" # fmt: skip
    assert all(layer.mlp.down_proj is not None for layer in model.model.layers), f"All layers must have a down_proj; model.model.layers={model.model.layers[0].mlp.down_proj}" # fmt: skip
    assert all(layer.mlp.down_proj.bias is not None for layer in model.model.layers), f"All layers must have a down_proj bias; model.model.layers={model.model.layers[0].mlp.down_proj.__dict__}" # fmt: skip
    assert all(layer.mlp.down_proj.bias.shape is not None for layer in model.model.layers), f"All layers must have a down_proj bias of shape (model_hidden_size,); model.model.layers={model.model.layers[0].mlp.down_proj.__dict__}" # fmt: skip
    assert all(layer.mlp.down_proj.bias.shape == (model_hidden_size,) for layer in model.model.layers), f"All layers must have a down_proj bias of shape (model_hidden_size,); model.model.layers={model.model.layers}" # fmt: skip
    sorted_layer_nums = sorted(layer_nums)
    # We try to make this as atomic as possible
    for i, layer_num in enumerate(sorted_layer_nums):
        steering_vec = steering_vecs[layer_num, relative_token_idx, :] # 0 => embedding, 1 => layer 1, 2 => layer 2 (index 1), etc...
        try:
            model.model.layers[layer_num - 2].mlp.down_proj.bias.data += scale * steering_vec
        except Exception as e:
            for j in range(i):
                layer_num = sorted_layer_nums[j]
                steering_vec = steering_vecs[layer_num, relative_token_idx, :]
                model.model.layers[layer_num - 2].mlp.down_proj.bias.data -= scale * steering_vec
            if allow_failure:
                return False
            else:
                raise e
            

    return True

def batched_generate_with_a_steered_model(
        model: LlamaForCausalLM,
        tokenizer: AutoTokenizer,
        datasets: Dict[str, Dataset],
        layer_nums_scale_rel_token_idx_groups: List[Tuple[List[int], int, int]],
        steering_vecs: torch.Tensor,
        allow_failure: bool = False,
        decoder_kwargs: Dict[str, Any] = {
            "batch_size": 511,
            "generation_kwargs": {
                "do_sample": True,
                "temperature": 1,
                "top_k": 50,
                "top_p": 0.95,
                "max_new_tokens": 2000,
            },
            "optimize_batch_size": False,
            "return_input_ids_pt": False,
            "return_optimal_batch_size": False,
            "return_activations": False,
            "device": None,
            "debug": True, # I'll gladly take some logspam hehe
        }
) -> Dict[str, BatchEncodedOutput]:
    """Apply steering vectors at some layer(s) of the model."""
    decoder = DatasetDecoder()
    dataset2output: Dict[str, BatchEncodedOutput] = {}
    if decoder_kwargs["device"] is None: # HOTFIX
        decoder_kwargs["device"] = model.device # HOTFIX
    n_failures = 0
    for i, (layer_nums, scale, relative_token_idx) in enumerate(
        tqdm.tqdm(
            layer_nums_scale_rel_token_idx_groups, desc="Applying steering vectors", total=len(layer_nums_scale_rel_token_idx_groups)
        )
    ): # fmt: skip
        applied_steering = False
        applied_steering = insert_steering_vector_into_model_bias(model, layer_nums, steering_vecs, scale, relative_token_idx)
        try:
            # NOTE: reuse the weights here
            for dataset_name, questions_dataset in datasets.items():
                output: BatchEncodedOutput = decoder.batch_decode(
                    model,
                    tokenizer,
                    questions_dataset,
                    **decoder_kwargs
                )
                # output = BatchEncodedOutput(generations=["test"]) # DEBUG
                dataset2output[dataset_name] = output
        except Exception as e:
            if allow_failure:
                print(f"Error applying steering vector at (iter={i}, layer_nums={layer_nums}): {e}")
                n_failures += 1
            else:
                raise e
        finally:
            # Make sure to remove the steering vector from the model (this must be atomic)
            if applied_steering:
                insert_steering_vector_into_model_bias(model, layer_nums, steering_vecs, -scale, relative_token_idx)
    return dataset2output, n_failures

def try_all_magnitudes_and_layers_steering(
        model: LlamaForCausalLM,
        tokenizer: AutoTokenizer,
        datasets: Dict[str, Dataset],
        magnitudes: List[float],
        layer_nums: List[int],
        steering_vecs: torch.Tensor,
        allow_failure: bool = False,
        decoder_kwargs: Dict[str, Any] = {
            "batch_size": 511,
            "generation_kwargs": {
                "do_sample": True,
                "temperature": 1,
                "top_k": 50,
                "top_p": 0.95,
                "max_new_tokens": 2000,
            },
            "optimize_batch_size": False,
            "return_input_ids_pt": False,
            "return_optimal_batch_size": False,
            "return_activations": False,
            "device": None,
            "debug": True, # I'll gladly take some logspam hehe
        },
        save_to_path: Optional[Path] = None
    ) -> Dict[Tuple[float, int], Dict[str, List[str]]]:
    """
    Try all combinations of magnitudes and layer numbers and return the results in a dictionary that is JSONable (
    and note that this will be just one layer at a time).
    """
    if allow_failure:
        raise NotImplementedError("Cannot allow failures here yet :P")
    combinations_to_try = list(itertools.product(magnitudes, layer_nums))
    mag_layer_num2dataset2output: Dict[Tuple[float, int], Dict[str, List[str]]] = {}
    for magnitude, layer_num in tqdm.tqdm(combinations_to_try, desc="Applying steering vectors", total=len(combinations_to_try)):
        print("="*100)
        print(f"Working with layer number = {layer_num}")
        print(f"Working with magnitude = {magnitude}")
        print("="*100)
        str_key = f"mag{magnitude}_layer{layer_num}"
        mag_layer_num2dataset2output[str_key], n_failures = batched_generate_with_a_steered_model(
            model,
            tokenizer,
            datasets,
            [([layer_num], magnitude, -1)], # On the last token
            steering_vecs,
            allow_failure=allow_failure, # Just logspam
            decoder_kwargs=decoder_kwargs # Default decoder kwargs defined above probably fine ngl
        )
        if n_failures > 0:
            raise RuntimeError(f"Failed to apply steering vector at layer_num={layer_num} with magnitude={magnitude}")
        # Make it immediatetly JSONable eh
        mag_layer_num2dataset2output[str_key] = {
            name: out.generations
            for name, out in mag_layer_num2dataset2output[str_key].items()
        }
        if save_to_path is not None:
            with open(save_to_path / f"{str_key}.json", "w") as f:
                json.dump(mag_layer_num2dataset2output[str_key], f, indent=4)
        assert all(isinstance(v, list) and all(isinstance(v2, str) for v2 in v) for v in mag_layer_num2dataset2output[str_key].values()) # fmt: skip
        print("="*100)
    return mag_layer_num2dataset2output

@click.command()
@click.option("--input-file", "-i", type=click.Path(exists=True), required=True)
@click.option("--steering-vectors-file", "-s", type=click.Path(exists=True), default="steering_vectors")
@click.option("--output-dir", "-o", type=click.Path(), default="steering_outputs")
def main(input_file: str, steering_vectors_file: str, output_dir: str):
    assert Path(steering_vectors_file).exists(), f"Steering vectors file does not exist: {steering_vectors_file}"
    assert not Path(output_dir).exists(), f"Output directory already exists: {output_dir}"
    assert Path(input_file).exists(), f"Input file does not exist: {input_file}"
    ################ 1. Load the launch config ################
    print("="*40 + " Loading launch config " + "="*40)
    with open(input_file, "r") as f:
        launch_config = json.load(f)
    magnitudes, layer_nums, device = launch_config["magnitudes"], launch_config["layer_nums"], launch_config["device"]
    assert isinstance(magnitudes, list) and isinstance(layer_nums, list) and isinstance(device, str)
    assert all(isinstance(magnitude, float) for magnitude in magnitudes)
    assert all(isinstance(layer_num, int) for layer_num in layer_nums)
    assert device.startswith("cuda:")

    ################ 2. Load the model ################
    print("="*40 + " Loading model " + "="*40)
    model_name = "meta-llama/Llama-3.2-1B-Instruct" # Small model to begin with
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # tokenizer.padding_side = "left" # I thought this was the right way to generate, but it makes the model suck; we use right side and it manages to .generate() and look OK somehow
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    add_zero_bias_to_all_layers_down_proj_bias(model)
    # print(model) # DEBUG
    model_hidden_size = model.model.embed_tokens.weight.data.shape[1]

    ################ 3. Load steering vectors ################
    print("="*40 + " Loading steering vectors " + "="*40)
    # XXX

    ################ 4. Load the datasets ################
    # NOTE: no dataset test or train custom for now :P
    print("="*40 + " Loading datasets " + "="*40)
    print("Creating rawwww HF datasets...")
    datasets_hf = {
        "reasoning": load_dataset("facebook/natural_reasoning", split="train").shuffle(seed=42).select(range(120)),
        "awesome": load_dataset("fka/awesome-chatgpt-prompts", split="train").shuffle(seed=42).select(range(120)),
        "gsm8k": load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(120)),
        "leetcode": load_dataset("greengerong/leetcode", split="train").shuffle(seed=42).select(range(120)),
    }
    print("Creating unique encoding args...")
    multi_encoding_args = EncodingArgs(
        # We WILL be doing full generation so let's be willing to go up to a reasonable length... I think llama was trained on
        # up to 8K or so tokens
        llm_tok_max_length=3000,
        llm_tok_max_prompt_length=1000,
        test_size_pct=0.1, # unfortunately, we will be just not using  the "test" set since ALL this is for testing LOL
        message_key={
            "reasoning": "question",
            "awesome": "prompt",
            "gsm8k": "question",
            "leetcode": "content",
        },
        # NOTE the usage of a new system prompt!
        # TODO(Adrianoh) allow for the creation of multiple different system prompts in this system
        system_prompt="""You are a helpful assistant tasked with helping users with a variety of tasks they may need help with, including, but not limited to:
        - Programming (i.e. you may recieve code or problem descriptions and you'll need to solve them)
        - Reasoning and mathematical problems (i.e. word problems, real world situations, etc...)
        - Technical, engineering, and scientific knowledge and problem solving
        - Business, legal, and financial knowledge and best-practices and planning
        - Information about documentation or other textual sources users may give you
        - General knowledge and trivia
        ...

        Make sure to be helpful, concise, and professional.
        """,
    )
    print("Creating (multi-dataset) encoder...")
    multi_dataset_encoder = DatasetEncoder(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets_hf,
        encoding_args=multi_encoding_args,
    )
    print("Encoding datasets...")
    tokenized_datasets = multi_dataset_encoder.get_trainable_datasets()
    assert set(tokenized_datasets.keys()) == set(datasets_hf.keys())
    datasets = {
        # Ignore this terminology, there is no training happening
        k: trainset_as_testset for k, (trainset_as_testset, _) in tokenized_datasets.items()
    }
    # assert "test_set" not in datasets
    assert all(isinstance(v, Dataset) for v in datasets.values())
    print("Printing dataset(s) information for debugging purposes...")
    for dataset_name, dataset in datasets.items():
        print(f"Dataset: {dataset_name}")
        print(dataset)
        print("=======")
        print(dataset[0])
        print("-"*100)

    ################ 5. Load the steering vectors ################
    print("="*40 + " Loading steering vectors " + "="*40)
    steering_vecs = safetensors_load_file(steering_vectors_file)["values"]
    assert steering_vecs.shape[0] == len(model.model.layers)+1, f"Steering vectors shape={steering_vecs.shape}"
    assert steering_vecs.shape[-1] == model_hidden_size, f"Steering vectors shape={steering_vecs.shape}"
    print(f"Steering vectors shape={steering_vecs.shape}")
    steering_vecs = steering_vecs.to(device)

    ################ 6. Run the steering experiments ################
    print("="*40 + " Running steering experiments " + "="*40)
    batched_steering_output_dir = Path(output_dir) / f"batched_steering_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    batched_steering_output_dir.mkdir(parents=True, exist_ok=False)
    assert len(list(batched_steering_output_dir.iterdir())) == 0, f"batched_steering_output_dir={batched_steering_output_dir} must be empty" # fmt: skip
    print("RUNNING THE ACTUAL EXPERIMENT OF STEERING (running the experiment)")
    all_ds = try_all_magnitudes_and_layers_steering(
        # Passthrough
        model,
        tokenizer,
        datasets,
        # Search space
        magnitudes,
        layer_nums,
        # Passthrough
        steering_vecs,
        allow_failure=False, # eh lmao
        # decoder_kwargs=decoder_kwargs, # Defaults are OK
        save_to_path=batched_steering_output_dir
    )
    # Quick hack just to be safe
    with open(str(uuid.uuid4()), "w") as f:
        json.dump(all_ds, f, indent=4)
    

if __name__ == "__main__":
    main()
