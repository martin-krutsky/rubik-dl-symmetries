# Config file describing move mappings of different symmetries

# nested dictionary of single ROTATION move mappings.
# - 1st lvl key describes the cube rotation direction:
#     U ↑   D ↓   L →   R ←   UL ↶   UR ↷
# - 2nd lvl describes the individual move mappings
rotation_gen_mapper = {
    'U': {'U': 'F', 'D': 'B', 'L': 'L', 'R': 'R', 'F': 'D', 'B': 'U'},
    'D': {'U': 'B', 'D': 'F', 'L': 'L', 'R': 'R', 'F': 'U', 'B': 'D'},
    'L': {'U': 'U', 'D': 'D', 'L': 'F', 'R': 'B', 'F': 'R', 'B': 'L'},
    'R': {'U': 'U', 'D': 'D', 'L': 'B', 'R': 'F', 'F': 'L', 'B': 'R'},
    'UL': {'U': 'R', 'D': 'L', 'L': 'U', 'R': 'D', 'F': 'F', 'B': 'B'},
    'UR': {'U': 'L', 'D': 'R', 'L': 'D', 'R': 'U', 'F': 'F', 'B': 'B'},
}

# for REFLECTION symmetry, we only need one dictionary/mapping
# other cube reflections can be achieved by composition of
# a rotation mapping and the reflection mapping below
reflection_gen_mapper = {
    'U': "U'", 'D': "D'", 'L': "R'", 'R': "L'", 'F': "F'", 'B': "B'"
}


# there are 24 rotations of cube, some of them cannot be achieved
# with a single rotation, but with a combination of them
rotation_symmetry_generators = [
    ['UL'], ['UL', 'UL'], ['UR'],
    ['U'], ['U', 'UL'], ['U', 'UL', 'UL'], ['U', 'UR'],
    ['D'], ['D', 'UL'], ['D', 'UL', 'UL'], ['D', 'UR'],
    ['L'], ['L', 'UL'], ['L', 'UL', 'UL'], ['L', 'UR'],
    ['R'], ['R', 'UL'], ['R', 'UL', 'UL'], ['R', 'UR'],
    ['U', 'U'], ['U', 'U', 'UL'], ['U', 'U', 'UL', 'UL'], ['U', 'U', 'UR'],
]

# action list ordered in a canonical way
actions = ["U'", "U", "D'", "D", "L'", "L", "R'", "R", "F'", "F", "B'", "B"]

# rotation dictionary for rotating single indices
rotation_dict = {
    'U': {
        0: 29, 1: 32, 2: 35, 3: 28, 4: 31, 5: 34, 6: 27, 7: 30, 8: 33, 
        9: 20, 10: 23, 11: 26, 12: 19, 13: 22, 14: 25, 15: 18, 16: 21, 17: 24, 
        18: 2, 19: 5, 20: 8, 21: 1, 22: 4, 23: 7, 24: 0, 25: 3, 26: 6,
        27: 11, 28: 14, 29: 17, 30: 10, 31: 13, 32: 16, 33: 9, 34: 12, 35: 15,
        36: 42, 37: 39, 38: 36, 39: 43, 40: 40, 41: 37, 42: 44, 43: 41, 44: 38,
        45: 47, 46: 50, 47: 53, 48: 46, 49: 49, 50: 52, 51: 45, 52: 48, 53: 51
    },
    'D': {
        0: 24, 1: 21, 2: 18, 3: 25, 4: 22, 5: 19, 6: 26, 7: 23, 8: 20, 
        9: 33, 10: 30, 11: 27, 12: 34, 13: 31, 14: 28, 15: 35, 16: 32, 17: 29, 
        18: 15, 19: 12, 20: 9, 21: 16, 22: 13, 23: 10, 24: 17, 25: 14, 26: 11,
        27: 6, 28: 3, 29: 0, 30: 7, 31: 4, 32: 1, 33: 8, 34: 5, 35: 2,
        36: 38, 37: 41, 38: 44, 39: 37, 40: 40, 41: 43, 42: 36, 43: 39, 44: 42,
        45: 51, 46: 48, 47: 45, 48: 52, 49: 49, 50: 46, 51: 53, 52: 50, 53: 47
    },
    
    'L': {
        0: 45, 1: 46, 2: 47, 3: 48, 4: 49, 5: 50, 6: 51, 7: 52, 8: 53, 
        9: 44, 10: 43, 11: 42, 12: 41, 13: 40, 14: 39, 15: 38, 16: 37, 17: 36,
        18: 20, 19: 23, 20: 26, 21: 19, 22: 22, 23: 25, 24: 18, 25: 21, 26: 24,
        27: 33, 28: 30, 29: 27, 30: 34, 31: 31, 32: 28, 33: 35, 34: 32, 35: 29,
        36: 8, 37: 7, 38: 6, 39: 5, 40: 4, 41: 3, 42: 2, 43: 1, 44: 0,
        45: 9, 46: 10, 47: 11, 48: 12, 49: 13, 50: 14, 51: 15, 52: 16, 53: 17
    },
    'R': {
        0: 44, 1: 43, 2: 42, 3: 41, 4: 40, 5: 39, 6: 38, 7: 37, 8: 36, 
        9: 45, 10: 46, 11: 47, 12: 48, 13: 49, 14: 50, 15: 51, 16: 52, 17: 53,
        18: 24, 19: 21, 20: 18, 21: 25, 22: 22, 23: 19, 24: 26, 25: 23, 26: 20,
        27: 29, 28: 32, 29: 35, 30: 28, 31: 31, 32: 34, 33: 27, 34: 30, 35: 33,
        36: 17, 37: 16, 38: 15, 39: 14, 40: 13, 41: 12, 42: 11, 43: 10, 44: 9,
        45: 0, 46: 1, 47: 2, 48: 3, 49: 4, 50: 5, 51: 6, 52: 7, 53: 8
    },
    'UL': {
        0: 2, 1: 5, 2: 8, 3: 1, 4: 4, 5: 7, 6: 0, 7: 3, 8: 6,
        9: 15, 10: 12, 11: 9, 12: 16, 13: 13, 14: 10, 15: 17, 16: 14, 17: 11,
        18: 36, 19: 37, 20: 38, 21: 39, 22: 40, 23: 41, 24: 42, 25: 43, 26: 44,
        27: 45, 28: 46, 29: 47, 30: 48, 31: 49, 32: 50, 33: 51, 34: 52, 35: 53,
        36: 27, 37: 28, 38: 29, 39: 30, 40: 31, 41: 32, 42: 33, 43: 34, 44: 35,
        45: 18, 46: 19, 47: 20, 48: 21, 49: 22, 50: 23, 51: 24, 52: 25, 53: 26
    },
    'UR': {
        0: 6, 1: 3, 2: 0, 3: 7, 4: 4, 5: 1, 6: 8, 7: 5, 8: 2,
        9: 11, 10: 14, 11: 17, 12: 10, 13: 13, 14: 16, 15: 9, 16: 12, 17: 15,
        18: 45, 19: 46, 20: 47, 21: 48, 22: 49, 23: 50, 24: 51, 25: 52, 26: 53,
        27: 36, 28: 37, 29: 38, 30: 39, 31: 40, 32: 41, 33: 42, 34: 43, 35: 44,
        36: 18, 37: 19, 38: 20, 39: 21, 40: 22, 41: 23, 42: 24, 43: 25, 44: 26,
        45: 27, 46: 28, 47: 29, 48: 30, 49: 31, 50: 32, 51: 33, 52: 34, 53: 35
    }
}

reflection_dict = {
    0: 11, 1: 10, 2: 9, 3: 14, 4: 13, 5: 12, 6: 17, 7: 16, 8: 15,
    9: 2, 10: 1, 11: 0, 12: 5, 13: 4, 14: 3, 15: 8, 16: 7, 17: 6, 
    18: 20, 19: 19, 20: 18, 21: 23, 22: 22, 23: 21, 24: 26, 25: 25, 26: 24,
    27: 29, 28: 28, 29: 27, 30: 32, 31: 31, 32: 30, 33: 35, 34: 34, 35: 33,
    36: 38, 37: 37, 38: 36, 39: 41, 40: 40, 41: 39, 42: 44, 43: 43, 44: 42,
    45: 47, 46: 46, 47: 45, 48: 50, 49: 49, 50: 48, 51: 53, 52: 52, 53: 51
}

pos_list = [
    #1
    [0.5, 0.5, 0], [1.5, 0.5, 0], [2.5, 0.5, 0],
    [0.5, 1.5, 0], [1.5, 1.5, 0], [2.5, 1.5, 0],
    [0.5, 2.5, 0], [1.5, 2.5, 0], [2.5, 2.5, 0],

    #2
    [2.5, 0.5, 3], [1.5, 0.5, 3], [0.5, 0.5, 3],
    [2.5, 1.5, 3], [1.5, 1.5, 3], [0.5, 1.5, 3],
    [2.5, 2.5, 3], [1.5, 2.5, 3], [0.5, 2.5, 3],

    #3
    [2.5, 0, 2.5], [2.5, 0, 1.5], [2.5, 0, 0.5],
    [1.5, 0, 2.5], [1.5, 0, 1.5], [1.5, 0, 0.5],
    [0.5, 0, 2.5], [0.5, 0, 1.5], [0.5, 0, 0.5],

    #4
    [0.5, 3, 2.5], [0.5, 3, 1.5], [0.5, 3, 0.5],
    [1.5, 3, 2.5], [1.5, 3, 1.5], [1.5, 3, 0.5],
    [2.5, 3, 2.5], [2.5, 3, 1.5], [2.5, 3, 0.5],

    #5
    [3, 2.5, 2.5], [3, 2.5, 1.5], [3, 2.5, 0.5],
    [3, 1.5, 2.5], [3, 1.5, 1.5], [3, 1.5, 0.5],
    [3, 0.5, 2.5], [3, 0.5, 1.5], [3, 0.5, 0.5],

    #6
    [0, 0.5, 2.5], [0, 0.5, 1.5], [0, 0.5, 0.5],
    [0, 1.5, 2.5], [0, 1.5, 1.5], [0, 1.5, 0.5],
    [0, 2.5, 2.5], [0, 2.5, 1.5], [0, 2.5, 0.5],
]

positions_on_cross_list = [
    [
    [-1, -1, -1,  27, 30, 33,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  28, 31, 34,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  29, 32, 35,  -1, -1, -1,  -1, -1, -1],

    [51, 52, 53,   6,  7,  8,  38, 37, 36,  15, 16, 17],
    [48, 49, 50,   3,  4,  5,  41, 40, 39,  12, 13, 14],
    [45, 46, 47,   0,  1,  2,  44, 43, 42,   9, 10, 11],

    [-1, -1, -1,  26, 23, 20,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  25, 22, 19,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  24, 21, 18,  -1, -1, -1,  -1, -1, -1],
    ],
    
    
    [
    [-1, -1, -1,  29, 28, 27,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  32, 31, 30,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  35, 34, 33,  -1, -1, -1,  -1, -1, -1],

    [ 6,  7,  8,  38, 37, 36,  15, 16, 17,  51, 52, 53],
    [ 3,  4,  5,  41, 40, 39,  12, 13, 14,  48, 49, 50],
    [ 0,  1,  2,  44, 43, 42,   9, 10, 11,  45, 46, 47],

    [-1, -1, -1,  20, 19, 18,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  23, 22, 21,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  26, 25, 24,  -1, -1, -1,  -1, -1, -1],
    ],
    
    
    [
    [-1, -1, -1,  35, 32, 29,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  34, 31, 28,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  33, 30, 27,  -1, -1, -1,  -1, -1, -1],

    [38, 37, 36,  15, 16, 17,  51, 52, 53,   6,  7,  8],
    [41, 40, 39,  12, 13, 14,  48, 49, 50,   3,  4,  5],
    [44, 43, 42,   9, 10, 11,  45, 46, 47,   0,  1,  2],

    [-1, -1, -1,  18, 21, 24,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  19, 22, 25,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  20, 23, 26,  -1, -1, -1,  -1, -1, -1],
    ],
    
    
    [
    [-1, -1, -1,  33, 34, 35,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  30, 31, 32,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  27, 28, 29,  -1, -1, -1,  -1, -1, -1],

    [15, 16, 17,  51, 52, 53,   6,  7,  8,  38, 37, 36],
    [12, 13, 14,  48, 49, 50,   3,  4,  5,  41, 40, 39],
    [ 9, 10, 11,  45, 46, 47,   0,  1,  2,  44, 43, 42],

    [-1, -1, -1,  24, 25, 26,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  21, 22, 23,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  18, 19, 20,  -1, -1, -1,  -1, -1, -1],
    ],
    
    
    [
    [-1, -1, -1,   6,  7,  8,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,   3,  4,  5,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,   0,  1,  2,  -1, -1, -1,  -1, -1, -1],

    [53, 50, 47,  26, 23, 20,  44, 41, 38,  35, 32, 29],
    [52, 49, 46,  25, 22, 19,  43, 40, 37,  34, 31, 28],
    [51, 48, 45,  24, 21, 18,  42, 39, 36,  33, 30, 27],
        
    [-1, -1, -1,  11, 10,  9,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  14, 13, 12,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  17, 16, 15,  -1, -1, -1,  -1, -1, -1],
    ],
    
    [
    [-1, -1, -1,  11, 10,  9,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  14, 13, 12,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,  17, 16, 15,  -1, -1, -1,  -1, -1, -1],

    [45, 48, 51,  27, 30, 33,  36, 39, 42,  18, 21, 24],
    [46, 49, 52,  28, 31, 34,  37, 40, 43,  19, 22, 25],
    [47, 50, 53,  29, 32, 35,  38, 41, 44,  20, 23, 26],

    [-1, -1, -1,   6,  7,  8,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,   3,  4,  5,  -1, -1, -1,  -1, -1, -1],
    [-1, -1, -1,   0,  1,  2,  -1, -1, -1,  -1, -1, -1],
    ]
]
