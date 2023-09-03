import itertools
import pickle

import numpy as np
import pandas as pd

from classes.cube_classes import Cube3State, Cube3
from generate.generate_states import generate_symmetric_cubes, ids_to_color, states_to_color
from generate.generate_random_states import generate_random_states_and_generators
from generate.symmetry_config import actions

from utils.compressions import plot_histo


X_MOVES = 6
DOUBLE_MOVES = False
DOUBLE_MOVES_NAME = 'double' if DOUBLE_MOVES else 'single'
cube_gens = [gen for i in range(1, X_MOVES + 1) for gen in itertools.product(actions, repeat=i)]

print('Number of generators', len(cube_gens))

def has_no_back_and_forth(x):
    for i in range(len(x)-1):
        if x[i] + "'" == x[i+1] or x[i] == x[i+1] + "'":
            return False
    return True

def has_three_in_row(x):
    for i in range(len(x)-2):
        if x[i] == x[i+1] and x[i] == x[i+2]:
            return False
    return True

cube_gens = list(filter(has_three_in_row, filter(has_no_back_and_forth, cube_gens)))
print(len(cube_gens))

generated_states, state_classes_list, state2gen_dict = generate_symmetric_cubes(cube_gens, double_moves=DOUBLE_MOVES)
print(generated_states[0][0].colors)

print('Number of generated states', len([cls_lis[1] for cls_lis in state_classes_list for state in cls_lis[0]]))

x_moves_dataset = pd.DataFrame({
    'state': [state for cls_list in generated_states for state in cls_list],
    'colors': [ids_to_color(state.colors) for cls_list in generated_states for state in cls_list],
    'class_id': [i for i, cls_list in enumerate(generated_states) for _ in cls_list],
    'distance': [cls_list[1] for cls_list in state_classes_list for _ in cls_list[0]],
    'generator': [state2gen_dict[" ".join(map(str, ids_to_color(state.colors)))] for cls_list in generated_states for state in cls_list]
})

zero_state = Cube3().generate_goal_states(1)[0]
print(x_moves_dataset[x_moves_dataset['state'] == zero_state])
x_moves_dataset = x_moves_dataset.drop(x_moves_dataset[x_moves_dataset['state'] == zero_state].index)
print(x_moves_dataset[x_moves_dataset['state'] == zero_state])
x_moves_dataset.to_csv(f'data/{X_MOVES}_moves_dataset_{DOUBLE_MOVES_NAME}.csv')
    
print(len(x_moves_dataset.index))