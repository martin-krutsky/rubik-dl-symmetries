import itertools
import pickle
import sys

import numpy as np
import pandas as pd

from classes.cube_classes import Cube3State, Cube3
from generate_states import generate_symmetric_cubes, ids_to_color, states_to_color, generate_states_and_generators
from symmetry_config import actions


NR_MOVES = int(sys.argv[1])
TAKE_DOUBLEMOVES = bool(int(sys.argv[2]))

cube_gens = []
for i in range(1, NR_MOVES+1):
    cube_gens += itertools.product(actions, repeat=i)

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
print('Number of generators after pruning', len(cube_gens))

generators, generated_states, state_classes_list = generate_symmetric_cubes(cube_gens, double_moves=TAKE_DOUBLEMOVES, min_iters=1000)

print('Number of generated states', len([cls_lis[1] for cls_lis in state_classes_list for state in cls_lis[0]]))

moves_dataset = pd.DataFrame({
    'state': [state for cls_lis in state_classes_list for state in cls_lis[0]],
    'colors': [state for cls_lis in state_classes_list for state in states_to_color(cls_lis[0])],
    'class_id': [i for i, cls_lis in enumerate(state_classes_list) for state in cls_lis[0]],
    'distance': [cls_lis[1] for cls_lis in state_classes_list for state in cls_lis[0]],
    'generator': [gen for cls_dict in generators for gen in [list(cls_dict['orig'])] + cls_dict['rotated']]
})
display(moves_dataset.head())

zero_state = Cube3().generate_goal_states(1)[0]
moves_dataset = moves_dataset.drop(moves_dataset[moves_dataset['state'] == zero_state].index)

if TAKE_DOUBLEMOVES:
    moves_dataset.to_pickle(f'data/generated_doublemoves/{NR_MOVES}_moves_dataset.pkl')
else:
    moves_dataset.to_pickle(f'data/generated_singlemoves/{NR_MOVES}_moves_dataset.pkl')
