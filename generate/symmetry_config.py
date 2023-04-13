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
