from dataclasses import dataclass


@dataclass
class Mapping:
    """
    2 tp groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    4 pp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]
    """

    world_size: int = 1
    rank: int = 0
    gpus_per_node: int = 8
    tp_size: int = 1
    pp_size: int = 1

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size
