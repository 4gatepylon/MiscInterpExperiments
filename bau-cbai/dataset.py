from __future__ import annotations
import pydantic
from pathlib import Path
from typing import List
import random
from jinja2 import Template

class Type2Object(pydantic.BaseModel):
    type2object: dict[str, list[str]]
    source: str

class DatasetItem(pydantic.BaseModel):
    type: str
    objects_ordering: List[str]
    correct_objects: List[str] # Subset of `objects_ordering`
    incorrect_objects: List[str] # Subset of `objects_ordering`

    def format_prompt(self, prompt_template: Template, as_openai_api: bool = False) -> str:
        string = prompt_template.render(
            type=self.type,
            objects_ordering=self.objects_ordering
        )
        if as_openai_api:
            return [
                {"role": "user", "content": string}
            ]
        else:
            return string

class Dataset(pydantic.BaseModel):
    type2object: Type2Object
    dataset: List[DatasetItem]

    def format_prompts(
            self,
            template_or_template_path: Path | str | Template = Path(__file__).parent / "prompt.j2",
            as_openai_api: bool = False
        ) -> List[str]:
        if not isinstance(template_or_template_path, Template):
            if isinstance(template_or_template_path, str):
                # String becomes a template if not a path else read from path
                template_path = Path(template_or_template_path)
                template_contents = template_path.read_text() if template_path.is_file() else template_or_template_path
            elif isinstance(template_or_template_path, Path):
                template_contents = template_or_template_path.read_text()
            else:
                raise ValueError(f"Invalid template_or_template_path: {template_or_template_path}")
            template = Template(template_contents)
        else:
            template = template_or_template_path
        assert isinstance(template, Template)
        return [
            item.format_prompt(as_openai_api=as_openai_api, prompt_template=template)
            for item in self.dataset
        ]

def load_type2object(path: Path | str = Path(__file__).parent / "type2object_gpt4o.json") -> Type2Object:
    path = Path(path)
    return Type2Object.model_validate_json(path.read_text())

def generate_dataset(
        type2object: Type2Object,
        n_examples: int = 5_000,
        n_objects_allowable: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9],
        n_correct_objects_allowable: List[int] = [1, 2, 3]) -> list[dict[str, list[str]]]:
    """
    Generate a dataset of type2object examples. Each one is picked as follows
    1. Pick a number of objects uniformly at random from the n_objects_allowable
    2. Pick a number of correct objects uniformly at random from the n_correct_objects_allowable 
        (that is SMALLER than the number of objects)
    3. Select a type uniformly at random from type2object.type2object.keys()
    4. Select the correct objects uniformly at random from the type2object.type2object[type]
    5. Select the incorrect objects uniformly at random from the type2object.type2object.values() excluding those from the type
    6. Pick a random ordering of the objects uniformly at random (i.e. shuffle the objects)
    7. Return the dataset
    """
    dataset = []
    # 1. Fill the set of values
    type2object_sets = {type: set(values) for type, values in type2object.type2object.items()}
    all_values_set = set()
    for values in type2object_sets.values():
        all_values_set |= values
    all_values_list = list(all_values_set)
    all_types_list = list(type2object.type2object.keys())
    # 2. Fill the dataset
    for _ in range(n_examples):
        type = random.choice(all_types_list)
        # [HEADER] Remove all these values from `all_values_set`
        sans_len = len(all_values_set)
        all_values_set -= type2object_sets[type]
        ################################################
        # Pick a number of correct + incorrect objects uniformly at random from the n_objects_allowable
        n_objects = random.choice(n_objects_allowable)
        n_correct_objects = random.choice([n for n in n_correct_objects_allowable if n <= n_objects])
        n_incorrect_objects = n_objects - n_correct_objects
        # Select the correct + incorrect objects uniformly at random from the type2object.type2object[type]
        # NOTE: we use the type2object.type2object[type] directly, not the type2object_sets[type] BECAUSE
        # it is a LIST (and that's where random sample is best); same for `all_values_list`
        correct_objects: List[str] = list(random.sample(type2object.type2object[type], n_correct_objects))
        incorrect_objects: List[str] = list(random.sample(all_values_list, n_incorrect_objects))
        # Pick a random ordering of the objects uniformly at random (i.e. shuffle the objects)
        objects_ordering: List = correct_objects + incorrect_objects
        random.shuffle(objects_ordering)
        # Add the item to the dataset
        dataset.append(DatasetItem(
            type=type,
            objects_ordering=objects_ordering,
            correct_objects=correct_objects,
            incorrect_objects=incorrect_objects
        ))
        ################################################
        # [FOOTER] Add all these values back to `all_values_set`
        all_values_set |= type2object_sets[type]
        assert len(all_values_set) == sans_len
    assert len(dataset) == n_examples
    return Dataset(type2object=type2object, dataset=dataset)