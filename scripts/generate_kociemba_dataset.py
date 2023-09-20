import re

import numpy as np
import pandas as pd

from generate.generate_states import generate_symmetric_cubes, ids_to_color

from utils.compressions import plot_histo

generators_kociemba, distances_kociemba = [], []

with open('data/kociemba100000optcubes.txt', 'r') as f:
    for i, line in enumerate(f):
        line = line.strip()
        generator_str, distance_raw = line.split('  ')
        distance = int(re.findall(r'\d+', distance_raw)[0])
        generator_str = re.sub(r"([A-Z])2(\')?", r"\1\2 \1\2", generator_str)
        generator = generator_str.split()

        generators_kociemba.append(generator)
        distances_kociemba.append(distance)

print('Number of generators', len(generators_kociemba))

generated_states_kociemba, state_classes_list_kociemba, state2gen_dict_kociemba = generate_symmetric_cubes(
    generators_kociemba, cube_gens_lengths=distances_kociemba)
print('Nr. of generated classes:', len(generated_states_kociemba))
print('Number of generated states:',
      len([cls_lis[1] for cls_lis in state_classes_list_kociemba for state in cls_lis[0]]))
print('Number of states in the first class:', len(generated_states_kociemba[0]))

kociemba_dataset = pd.DataFrame({
    'state': [state for cls_list in generated_states_kociemba for state in cls_list],
    'colors': [ids_to_color(state.colors) for cls_list in generated_states_kociemba for state in cls_list],
    'class_id': [i for i, cls_list in enumerate(generated_states_kociemba) for _ in cls_list],
    'distance': [cls_list[1] for cls_list in state_classes_list_kociemba for _ in cls_list[0]],
    'generator': [state2gen_dict_kociemba[" ".join(map(str, ids_to_color(state.colors)))] for cls_list in
                  generated_states_kociemba for state in cls_list]
})

kociemba_dataset.to_csv(f'data/processed/kociemba_dataset.csv')

class_compressions_kociemba = list(map(len, generated_states_kociemba))
plot_histo(class_compressions_kociemba, f'imgs/dataset_visualizations/kociemba/all_kociemba_class_sizes_histo.png')

for move_nr in sorted(list(set(distances_kociemba))):
    filtered_gen_states = np.array(generated_states_kociemba, dtype=object)[
        (np.array(list(map(lambda x: x[1], state_classes_list_kociemba)), dtype=object) == int(move_nr))]
    print(f'{move_nr} move(s) from goal')
    filtered_class_compressions = list(map(len, filtered_gen_states))
    plot_histo(filtered_class_compressions,
               f'imgs/dataset_visualizations/kociemba/kociemba_{move_nr}moves_class_sizes_histo.png')
    print()
