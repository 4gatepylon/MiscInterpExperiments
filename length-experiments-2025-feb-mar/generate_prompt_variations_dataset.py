from __future__ import annotations
import click
from pathlib import Path
import numpy as np
import json
import yaml
import jinja2
import pydantic
import litellm
import openai
import tqdm
from typing import Optional, List, Dict, Union, Any

class DatasetCreationPipeline:
    """
    A simple class to abstract away the creation of a length analysis dataset. 
    
    It handles the following (that happens roughly in this order):
        1. Generating prompts from templates and combinations of them with values that are meant to fill in the template slots.
        2. Generating variations on prompts using a larger LLM (i.e. GPT or Claude or something else)
        3. Generating random typos to make the prompts use more realistic (NOTE: uses: https://github.com/ranvijaykumar/typo and you will
            want to git clone somewhere and then pip3 install -e . when you are in the same conda/virtualenv environ.)
        4. Saving and loading these prompts for LLM inference
        5. Doing LLM (batch) inference with some common LOCAL LLMs (assuming that you have access to some sort of cuda) and saving the
            outputs alongside the inputs. Allows for common generation parameters and can also store activations.
        6. The ability to load the above outputs as a huggingface dataset.
    
    Basically this enables you to easily get a dataset that you can treat as a regular machine learning dataset for analysis of whether or not
    length can be predicted (etc...). However, for STEERING experiments you will still need an LLM in the loop.
    """

    @staticmethod
    def generate_random_typos(string: str, num_typos: int, seed: Optional[int] = None, use_tqdm: bool = False) -> str:
        num_typos_target = num_typos # imo nicer names like this at each level of abstraction
        rng = np.random.default_rng(seed)
        seed = int(rng.integers(2**32))
        type_function_kwargs_tuple_list_str = [
            # Char Swap
            ("typo.StrErrer", "char_swap", {}),
            ("typo.StrErrer", "char_swap", {"preservefirst": True}),
            ("typo.StrErrer", "char_swap", {"preservelast": True}),
            # Missing Char
            ("typo.StrErrer", "missing_char", {}),
            ("typo.StrErrer", "missing_char", {"preservefirst": True}),
            ("typo.StrErrer", "missing_char", {"preservelast": True}),
            # Extra Char
            ("typo.StrErrer", "extra_char", {}),
            ("typo.StrErrer", "extra_char", {"preservefirst": True}),
            ("typo.StrErrer", "extra_char", {"preservelast": True}),
            # Nearby Char
            ("typo.StrErrer", "nearby_char", {}),
            ("typo.StrErrer", "nearby_char", {"preservefirst": True}),
            ("typo.StrErrer", "nearby_char", {"preservelast": True}),
            # Other String Errors
            ("typo.StrErrer", "similar_char", {}),
            ("typo.StrErrer", "skipped_space", {}),
            ("typo.StrErrer", "random_space", {}),
            ("typo.StrErrer", "repeated_char", {}),
            ("typo.StrErrer", "unichar", {}),
        ]
        type_function_kwargs_tuple_list_int = [
            # Integer Errors
            ("typo.IntErrer", "digit_swap", {}),
            ("typo.IntErrer", "digit_swap", {"preservefirst": True}),
            ("typo.IntErrer", "digit_swap", {"preservelast": True}),
            ("typo.IntErrer", "missing_digit", {}),
            ("typo.IntErrer", "missing_digit", {"preservefirst": True}),
            ("typo.IntErrer", "missing_digit", {"preservelast": True}),
            ("typo.IntErrer", "extra_digit", {}),
            ("typo.IntErrer", "extra_digit", {"preservefirst": True}),
            ("typo.IntErrer", "extra_digit", {"preservelast": True}),
            ("typo.IntErrer", "nearby_digit", {}),
            ("typo.IntErrer", "nearby_digit", {"preservefirst": True}),
            ("typo.IntErrer", "nearby_digit", {"preservelast": True}),
            ("typo.IntErrer", "similar_digit", {}),
            ("typo.IntErrer", "repeated_digit", {}),
            ("typo.IntErrer", "unidigit", {}),
        ]
        type_function_kwargs_tuple_list_date = [
            # Datetime Errors
            ("typo.DateErrer", "date_month_swap", {})
        ]
        type_name2allow = { "typo.StrErrer": True, "typo.IntErrer": True, "typo.DateErrer": True } # fmt: skip
        type_name2function_kwargs_tuple_list = { "typo.StrErrer": type_function_kwargs_tuple_list_str, "typo.IntErrer": type_function_kwargs_tuple_list_int, "typo.DateErrer": type_function_kwargs_tuple_list_date } # fmt: skip
        type_name2type = { "typo.StrErrer": typo.StrErrer, "typo.IntErrer": typo.IntErrer, "typo.DateErrer": typo.DateErrer } # fmt: skip
        type_function_kwargs_tuple_list = type_function_kwargs_tuple_list_str + type_function_kwargs_tuple_list_int + type_function_kwargs_tuple_list_date # fmt: skip
        tqdm_func = tqdm.tqdm if use_tqdm else lambda x, *args, **kwargs: x
        num_typos_made = 0
        num_throws = 0
        for _ in tqdm_func(range(num_typos_target)):
            # Call a function by name using getattr
            thrown_error = True
            num_typos_added = 0
            while thrown_error:
                try:
                    obj_type_name, function_name, function_kwargs = random.choice(type_function_kwargs_tuple_list) # fmt: skip
                    obj_type = type_name2type[obj_type_name]
                    obj = obj_type(string, seed=seed)
                    string = getattr(obj, function_name)(**function_kwargs).result
                    seed = int(rng.integers(2**32))
                    thrown_error = False
                    num_typos_made += 1
                    num_typos_added += 1
                except Exception as __:
                    num_throws += 1
                    assert num_throws <= 2, f"num_throws={num_throws}"
                    assert type_name2allow[obj_type_name]
                    type_name2allow[obj_type_name] = False
                    prev_len = len(type_function_kwargs_tuple_list)
                    type_function_kwargs_tuple_list = []
                    # This will be called at most twice (two remove two types)
                    # since at the very least string typos will work
                    for type_name, allow in type_name2allow.items():
                        if allow:
                            type_function_kwargs_tuple_list += type_name2function_kwargs_tuple_list[type_name] # fmt: skip
                    assert len(type_function_kwargs_tuple_list) < prev_len, f"len(type_function_kwargs_tuple_list)={len(type_function_kwargs_tuple_list)}, prev_len={prev_len}" # fmt: skip
            assert num_typos_added == 1, f"num_typos_added={num_typos_added}"
        assert num_typos_made == num_typos_target, f"num_typos_made={num_typos_made}, num_typos_target={num_typos_target}"
        return string

    def __init__(self) -> None:
        pass

def api_generate(
    prompts: Union[List[str], List[List[Dict[str, str]]]],
    model: str,
    num_retries: int = 4,
    batch_size: int = 16,
    max_new_tokens=128,
    tqdm_enabled: bool = False,  # Nawwww
    completion_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str | Any]: # <--- could be any because we are possibly asking for the utilization of JSON mode
    """
    This is a helper function to make it easy to generate using various LLM APIs
    (e.g. OpenAI, Anthropic, etc.) with built in error-handling.

    prompts can be either a list of string prompts, or it can be a list of multi-turn
    conversations in huggingface format:
        [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response},
            {"role": "user", "content": user_input1},
            ...
        ]
    
    Returns:
        - USUALLY (this is probably you): a list of the response strongs.
        - Sometimes a JSON object or other type of object if you request since you can send arbitrary kwargs.
            - https://docs.litellm.ai/docs/completion/json_mode
            - https://platform.openai.com/docs/guides/structured-outputs
    """

    # If we pass a list of prompts, convert to message format
    if isinstance(prompts[0], str):
        prompts = [[{"role": "user", "content": p}] for p in prompts]

    try:
        # Attempt batched completion call with litellm
        responses = []
        for i in range(0, len(prompts), batch_size):
            r = litellm.batch_completion(
                model=model,
                messages=prompts[i : i + batch_size],
                max_tokens=max_new_tokens,
                num_retries=num_retries,
                **completion_kwargs,
            )
            responses.extend(r)
        new_texts = [r.choices[0].message.content for r in responses]

    except openai.OpenAIError as e:
        # Error handling
        should_retry = litellm._should_retry(e.status_code)
        print("Error: API failed to respond.", e, f"should_retry: {should_retry}")
        new_texts = []

    return new_texts

class ExchangeEntry(pydantic.BaseModel):
    """
    Entry to encode information about a single exchange with a language model. Each exchange is basically a single prompt
    and a single response. The exchange encoes information about:
    - What the prompt was and any sort of prompt generation parameters (since we are creating variations on prompts, etc...)
    - What the response was and any generation parameters
    """

    # Key information from the creation of the template
    topic: str
    length: int
    templated_prompt: str

    # Information about the variation
    variation_prompt: Optional[str] = None
    variation_idx: int = 0  # 0 if NOT a variation and 1 otherwise

    # Possible response information from the LLM (in the future we will )
    response: Optional[str] = None
    response_temperature: Optional[float] = None # should be between 0 and infinity; if it's 0 this means "greedy"
    response_top_p: Optional[float] = None
    response_top_k: Optional[int] = None
    response_max_tokens: Optional[int] = None

    @staticmethod
    def is_pure_template(prompt_entry: ExchangeEntry) -> bool:
        """
        A pure template is one that has not been varied.
        """
        assert prompt_entry.variation_idx >= 0
        assert (prompt_entry.variation_prompt is None) == (prompt_entry.variation_idx == 0)
        return prompt_entry.variation_prompt is None


@click.command()
@click.option("--topics-file", "-t", type=click.Path(exists=True), required=False, default="generate_prompt_variations_dataset/default_topics.json")  # fmt: skip
@click.option("--output-file", "-o", type=click.Path(), required=True, default="generate_prompt_variations_dataset/output.json")  # fmt: skip
@click.option("--clobber-output-file", "-clobber", "-clob", is_flag=True, help="Clobber the output file if it exists")  # fmt: skip
# Template options
@click.option("--template-topic-key", "-ttk", type=str, required=False, default="topic")  # fmt: skip
@click.option("--template-length-key", "-tlk", type=str, required=False, default="length")  # fmt: skip
@click.option("--template-length-options-file", "-tt", type=str, required=False, default="generate_prompt_variations_dataset/default_length_options.json")  # fmt: skip
@click.option("--template-file", "-p", type=click.Path(exists=True), required=True, default="generate_prompt_variations_dataset/default_template.txt")  # fmt: skip
# Prompt engineering options
@click.option("--use-variation", "-uv", is_flag=True, help="Use a variation prompt to generate variations. Basically, it will use your prompt to ")  # fmt: skip
@click.option("--variation-model", "-vm", type=str, required=False, default="gpt-4o-mini")  # fmt: skip
@click.option("--variation-template", "-vt", type=str, required=False, default="generate_prompt_variations_dataset/default_variation_template.txt")  # fmt: skip
@click.option("--variation-template-num-variations-key", "-vtnk", type=str, required=False, default="num_variations")  # fmt: skip
@click.option("--variation-template-prompt-key", "-vtpk", type=str, required=False, default="prompt")  # fmt: skip
@click.option("--num-variations", "-nv", type=int, required=False, default=5)  # fmt: skip
def main(
    topics_file: str,
    output_file: str,
    clobber_output_file: bool,
    template_topic_key: str,
    template_length_key: str,
    template_length_options_file: str,
    template_file: str,
    use_variation: bool,
    variation_template: str,
    variation_template_num_variations_key: str,
    variation_template_prompt_key: str,
    num_variations: int,
) -> None:
    """
    Quick script to generate a static dataset (file) that can be used to generate different lengths' prompts. It is pretty simple. Does the following:
    1. Generate a bunch of prompts with different length output requests. Every prompt has roughly the same format (if you use defaults):
        ```
        something something create me {{ some number }} of sentences/paragraphs/etc... about {{ some topic }}
        ```
    2. For each prompt generate a bunch of variations of the same prompt using a larger LLM (i.e. GPT or Claude or something else)
        TODO(Adriano) 
    3. Save the prompt to an output file in a way that can easily be formatted into a Huggingface Dataset. These will always include the following:
        - prompt (the prompt requested by the user)
        - length (the length of the response requested by the user)
        - variation (the variation of the prompt requested by the user)
    """
    # 0. Sanity checking
    assert not Path(output_file).parent.exists()
    # 1. Load topics + Template
    topics = json.loads(Path(topics_file).read_text())
    assert isinstance(topics, list) and all(
        isinstance(topic, str) for topic in topics
    ), f"{type(topics)} & {type(topics[0]) if len(topics) > 0 else ''}"
    template_content = Path(template_file).read_text()
    template_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(template_file).parent),
    )
    template = template_env.from_string(template_content)
    rendering = template.render(topic=topics[0], length=5)
    print(rendering)
    raise NotImplementedError("Not implemented")  # XXX

    # # Load template
    # template_file = click.prompt(
    #     "Enter template file path", type=click.Path(exists=True)
    # )
    # try:
    #     with open(template_file, "r") as f:
    #         template_content = f.read()

    #     # Set up Jinja environment
    #     template_env = jinja2.Environment(
    #         loader=jinja2.FileSystemLoader(Path(template_file).parent),
    #         autoescape=jinja2.select_autoescape(["html", "xml"]),
    #     )
    #     template = template_env.from_string(template_content)

    #     # Generate variations using the template
    #     variations = []
    #     for topic in topics:
    #         rendered = template.render(topic=topic)
    #         variations.append(rendered)

    #     # Write to output file
    #     with open(output_file, "w") as f:
    #         json.dump(variations, f, indent=2)

    #     print(
    #         f"Successfully generated {len(variations)} variations and saved to {output_file}"
    #     )

    # except Exception as e:
    #     print(f"Error processing template: {e}")


if __name__ == "__main__":
    main()
