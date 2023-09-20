from typing import List, Tuple, Any
from random import randrange

import numpy as np

from classes.cube_classes import Cube3State, Cube3
from .generate_states import move_index_to_char


def generate_random_states_and_generators(
        num_states: int, backwards_range: Tuple[int, int]
) -> Tuple[List[Cube3State], List[int], List[List[Any]]]:
    """
    Generate `num_states` of states, by performing random scrambles on the solved states, where the number of scrambles
    is in the `backwards_range`, inclusive. The resulting distribution is biased towards states closer to the solved
    state, as we use distribution uniform w.r.t. that distance, but there are less unique states as we get closer to
    the solution (the state space is a tree with root in the solved state).

    The code is taken (and adjusted) from the https://github.com/forestagostinelli/DeepCubeA repository
    """
    # Initialize
    cube = Cube3()
    scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
    num_env_moves: int = cube.get_num_moves()

    # Get goal states
    states_np: np.ndarray = cube.generate_goal_states(num_states, np_format=True)

    # Scrambles
    scramble_nums: np.array = np.random.choice(scrambs, num_states)
    generators = [[] for _ in range(num_states)]
    num_back_moves: np.array = np.zeros(num_states)

    # Go backward from goal state
    moves_lt = num_back_moves < scramble_nums
    while np.any(moves_lt):
        idxs: np.ndarray = np.where(moves_lt)[0]
        subset_size: int = int(max(len(idxs) / num_env_moves, 1))
        idxs: np.ndarray = np.random.choice(idxs, subset_size)

        move: int = randrange(num_env_moves)
        char_move: str = move_index_to_char(move)
        states_np[idxs], _ = cube._move_np(states_np[idxs], move)
        for idx in idxs:
            generators[idx].append(char_move)

        num_back_moves[idxs] = num_back_moves[idxs] + 1
        moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

    states: List[Cube3State] = [Cube3State(x) for x in list(states_np)]

    return states, scramble_nums.tolist(), generators
