from functools import lru_cache
import pickle
from typing import Dict, List, Tuple
from random import randrange

import numpy as np
import pandas as pd
from tqdm import tqdm

from classes.cube_classes import Cube3State, Cube3
from .symmetry_config import rotation_gen_mapper, rotation_symmetry_generators, reflection_gen_mapper


# --- UTILS ---

@lru_cache(maxsize=None)
def char_to_move_index(char: str) -> int:
    '''Translate a move string into its index.'''
    CHAR_MOVE_DICT = {
        "U'": 0, "U": 1, "D'": 2, "D": 3, "L'": 4, "L": 5,
        "R'": 6, "R": 7, "F'": 8, "F": 9, "B'": 10, "B": 11
    }
    return CHAR_MOVE_DICT[char]


@lru_cache(maxsize=None)
def move_index_to_char(move: int) -> str:
    '''Translate a move index into move string.'''
    MOVE_CHAR_DICT = {
        0: "U'", 1: "U", 2: "D'", 3: "D", 4: "L'", 5: "L",
        6: "R'", 7: "R", 8: "F'", 9: "F", 10: "B'", 11: "B"
    }
    return MOVE_CHAR_DICT[move]


def id_to_color(index: int) -> int:
    '''Convert cubie index to color index.'''
    return index // 9


def ids_to_color(id_list: np.array) -> List[int]:
    '''Convert an array of cubie indices to color indices.'''
    return list(map(id_to_color, id_list))


def states_to_color(state_list: np.array) -> np.array:
    '''Convert multiple states (arrays of indices) to color indices arrays.'''
    return np.array(list(map(lambda x: ids_to_color(x.colors), state_list)))

# --- UTILS ---


# --- EXTENDING SYMMETRY DICTIONARIES ---

def extend_with_counterclockwise(move_mapper_clockwise: Dict[str, str], check_doublequote: bool = False) -> Dict[str, str]:
    '''
    The dictionaries in `symmetry_config` define only half of the mappings.
    Their counterclockwise counterparts are generated here.
    '''
    move_mapper_extended = {
        key + n: move_mapper_clockwise[key] + n 
        for key in move_mapper_clockwise
        for n in ["", "'"]
    }
    if check_doublequote:
        for key in move_mapper_extended:
            if len(move_mapper_extended[key]) == 3:
                move_mapper_extended[key] = move_mapper_extended[key][0] 
    return move_mapper_extended


@lru_cache(maxsize=None)
def get_rotation_move_mapper(direction: str) -> Dict[str, str]:
    '''
    a wrapper for ROTATION dictionary defined in `symmetry_config` including the counterclokwise mappings
    '''
    move_mapper_clockwise = rotation_gen_mapper.get(direction, None)
    if move_mapper_clockwise is None:
        raise ValueError('Wrong rotation string!')

    move_mapper = extend_with_counterclockwise(move_mapper_clockwise)
    return move_mapper

# --- EXTENDING SYMMETRY DICTIONARIES ---



# --- GENERATOR TRANSFORMATION GIVEN SPECIFIC ROTATION, REFLECTION ---

def rotate_generator(cube_generator: List[str], direction: str) -> List[str]:
    '''
    cube_generator: list of moves from the solved state leading to a cube
    direction: direction of the rotation
        U - up, D - down, L - left, R - right
        UL - upper to left, UR - upper to right
    '''
    move_mapper = get_rotation_move_mapper(direction)
    return [move_mapper[key] for key in cube_generator]


def reflect_generator(cube_generator: List[str]) -> List[str]:
    '''
    cube_generator: list of moves from the solved state leading to a cube
    '''
    
    move_mapper = extend_with_counterclockwise(reflection_gen_mapper, check_doublequote=True)
    return [move_mapper[key] for key in cube_generator]


def rotate_generator_by_sequence(cube_generator: List[str], rotation_symmetry: str) -> List[str]:
    '''Select the right mapping dictionary based on the rotation string.'''
    for action in rotation_symmetry:
        cube_generator = rotate_generator(cube_generator, action)
    return cube_generator

# --- GENERATOR TRANSFORMATION GIVEN SPECIFIC ROTATION, REFLECTION ---


# --- CUBE SCRAMBLING ---

def generate_cubestate(cube_generator: List[str]) -> Cube3State:
    '''
    cube_generators: list of move sequences, each sequence represents a cube
    Generate a cube scrambled according to the generator
    '''
    cube = Cube3()
    curr_state = cube.generate_goal_states(1)[0]
    for operation in cube_generator:
        # next_state_ls, _ = cube.next_state([curr_state], char_to_move_index(operation))
        prev_state_ls = cube.prev_state([curr_state], char_to_move_index(operation))
        curr_state = prev_state_ls[0]
    return curr_state

# --- CUBE SCRAMBLING ---


# --- GENERATING SYMMETRIC CUBES ---

def generate_rotated_cubestates(cube_generator: List[str], orig_state: Cube3State) -> Tuple[List[List[str]], List[Cube3State]]:
    '''
    cube_generator: a move sequence representing the original cube
    Generate a list of cubes rotated in all possible ways and then scrambled according to the generator.
    '''
    generators = [cube_generator]
    states = [orig_state]
    gen_set = {tuple(cube_generator)}
    state_set = {orig_state.colors.tostring()}
    for i, rotation_symmetry in enumerate(rotation_symmetry_generators):
        rotation_generator = rotate_generator_by_sequence(cube_generator, rotation_symmetry)

        if tuple(rotation_generator) in gen_set:
            continue
        rotated_state = generate_cubestate(rotation_generator)
        if rotated_state.colors.tostring() in state_set:
            continue
        gen_set.add(tuple(rotation_generator))
        state_set.add(rotated_state.colors.tostring())
        
        generators.append(rotation_generator)
        states.append(rotated_state)
    return generators, states


def generate_reflected_cubestate(cube_generator: List[str], orig_state: Cube3State) -> Tuple[List[str], Cube3State]:
    '''
    cube_generator: a move sequence representing the original cube
    Generate a list of cubes reflected and then scrambled according to the generator.
    '''
    orig_state_str = orig_state.colors.tostring()
    reflection_generator = reflect_generator(cube_generator)

    if tuple(reflection_generator) == tuple(cube_generator):
        return None, None
    reflected_state = generate_cubestate(reflection_generator)
    if reflected_state.colors.tostring() == orig_state_str:
        return None, None

    return reflection_generator, reflected_state


def generate_inverse_cubestates(cube_generators: List[List[str]], orig_state: Cube3State) -> Tuple[List[List[str]], List[Cube3State]]:
    '''
    cube_generators: list of move sequences, each sequence represents a cube
    Generate a list of cubes generated by inverting the cube generator.
    '''
    inversed_generators, inversed_states = [], []
    for cube_generator, orig_state in zip(cube_generators, orig_states):
        orig_state_str = orig_state.colors.tostring()
        inversed_generator = list(reversed(list(map(lambda x: x[:-1] if x[-1] == "'" else x + "'", cube_generator))))

        if tuple(inversed_generator) == tuple(cube_generator):
            continue
        inversed_state = generate_cubestate(inversed_generator)
        if inversed_state.colors.tostring() == orig_state_str:
            continue
        inversed_generators.append(inversed_generator)
        inversed_states.append(inversed_state)
    return inversed_generators, inversed_states

# --- GENERATING SYMMETRIC CUBES ---


def calculate_length_of_generators(cube_generators: List[List[str]], double_moves_as_one: bool = True) -> List[int]:
    '''
    Calculate the lengths of `cube_generators` with the possibility to (not) count double moves as one.
    '''
    cube_gens_length = []
    for generator in cube_generators:
        gen_len = len(generator)
        if double_moves_as_one:
            notrem_flag = True
            for i in range(1, len(generator)):
                if notrem_flag and generator[i-1] == generator[i]:
                    gen_len -= 1
                    notrem_flag = False
                else:
                    notrem_flag = True
        cube_gens_length.append(gen_len)
    return cube_gens_length


def generate_symmetric_cubes(
        cube_generators: List[List[str]], double_moves: bool = False, mininterval: int = 1, add_inverse: bool = False
    ) -> Tuple[List[List[Cube3State]], Tuple[List[List[Cube3State]], int], Dict[str, List[List[str]]]]:
    '''
    Putting together generation of all symmetries from a dataset of generators `cube_generators`.
    - double_moves, bool - if True, count double_moves as one move
    - mininterval, int - used 
    '''
    cube_gens_length = calculate_length_of_generators(cube_generators, double_moves_as_one=double_moves)
    
    state2gen_dict = dict()
    generated_states = []
    state_classes_list = []
    for i, cube_gen in enumerate(tqdm(cube_generators, mininterval=mininterval)):
        orig_state = generate_cubestate(cube_gen)
        state_clr_str = " ".join(map(str, ids_to_color(orig_state.colors)))
        # state_clr_str = orig_state.colors.tostring()
        if state_clr_str in state2gen_dict:
            continue
        
        rotated_geners, rotated_states = generate_rotated_cubestates(cube_gen, orig_state)
        ref_gen, ref_state = generate_reflected_cubestate(cube_gen, orig_state)
        if ref_gen is not None and  ref_state.colors.tostring() not in state2gen_dict:
            reflected_geners, reflected_states = generate_rotated_cubestates(ref_gen, ref_state)
        else:
            reflected_geners, reflected_states = [], []
        all_geners, all_states = rotated_geners + reflected_geners, rotated_states + reflected_states
        
        if add_inverse:
            inversed_geners, inversed_states = generate_inverse_cubestates(all_geners, all_states)
            all_geners, all_states = rotated_reflected_geners + inversed_geners, rotated_reflected_states + inversed_states
        
        states_to_append = []
        for j, (generator, state) in enumerate(zip(all_geners, all_states)):
            state_clr_str = " ".join(map(str, ids_to_color(state.colors)))
            # state_clr_str = state.colors.tostring()
            if state_clr_str in state2gen_dict:
                state2gen_dict[state_clr_str].append(generator)
            else:
                state2gen_dict[state_clr_str] = [generator]
                states_to_append.append(state)
        generated_states.append(states_to_append)
        state_classes_list.append((states_to_append, cube_gens_length[i]))  # len(cube_gen)))
    return generated_states, state_classes_list, state2gen_dict
