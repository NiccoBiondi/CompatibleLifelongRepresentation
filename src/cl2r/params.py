import os.path as osp
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentParams(): 
    root_folder: str = './'

    epochs: int = 70
    stages: List[int] = field(default_factory=lambda: [49, 64])
    batch_size: int = 128
    num_workers: int = 8

    start: int = 50
    increment: int = 10

    criterion_weight_base: float = 10
    rehearsal: int = 20

    save_best: bool = False

def save_default_yaml(path):
    # -*- coding: utf-8 -*-
    import yaml
    import io

    data = ExperimentParams().__dict__

    if not path.split(".")[-1] == "yml":
        path = path + '.yml'

    # Write YAML file
    with io.open(path, 'w', encoding='utf8') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Read YAML file
    with open(path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    print(data == data_loaded)


if __name__ == "__main__":
    path = osp.join(osp.abspath(osp.curdir), 'config.yml')
    save_default_yaml(path)
