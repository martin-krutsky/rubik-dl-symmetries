import itertools
import pickle
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from classes.cube_classes import Cube3State, Cube3
from generate.generate_states import generate_symmetric_cubes, ids_to_color, states_to_color
from generate.generate_random_states import generate_random_states_and_generators
from generate.symmetry_config import actions

from utils.compressions import plot_histo


generators_kociemba, distances_kociemba = [], []

with open('data/kociemba100000optcubes.txt', 'r') as f:
    for i, line in tqdm(enumerate(f)):
        line = line.strip()
        generator_str, distance_raw = line.split('  ')
        distance = int(re.findall(r'\d+', distance_raw)[0])
        generator_str = re.sub(r"([A-Z])2(\')?", r"\1\2 \1\2", generator_str)
        generator = generator_str.split()
        
        generators_kociemba.append(generator)
        distances_kociemba.append(distance)
        
np.random.seed(42)
KOCIEMBAS_SIZE = 10000
randomly_chosen = np.random.choice(len(distances_kociemba), size=KOCIEMBAS_SIZE, replace=False)
generators_kociemba_subset = np.array(generators_kociemba)[randomly_chosen]
distances_kociemba_subset = np.array(distances_kociemba)[randomly_chosen]
print(distances_kociemba_subset)

generated_states_kociemba, state_classes_list_kociemba, state2gen_dict_kociemba = generate_symmetric_cubes(generators_kociemba_subset, cube_gens_lengths=distances_kociemba_subset)
print(generated_states_kociemba[0][0].colors)

kociemba_dataset = pd.DataFrame({
    'state': [state for cls_list in generated_states_kociemba for state in cls_list],
    'colors': [ids_to_color(state.colors) for cls_list in generated_states_kociemba for state in cls_list],
    'class_id': [i for i, cls_list in enumerate(generated_states_kociemba) for _ in cls_list],
    'distance': [cls_list[1] for cls_list in state_classes_list_kociemba for _ in cls_list[0]],
    'generator': [state2gen_dict_kociemba[" ".join(map(str, ids_to_color(state.colors)))] for cls_list in generated_states_kociemba for state in cls_list]
})
print(kociemba_dataset.head())

kociemba_dataset.to_csv(f'data/processed/kociemba{KOCIEMBAS_SIZE}_dataset.csv')
kociemba_dataset.to_pickle(f'data/processed/kociemba{KOCIEMBAS_SIZE}_dataset.pkl')