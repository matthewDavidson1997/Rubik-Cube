import copy
import random

import pyvista as pv
import numpy as np


MOVES = {"U", "D", "F", "B", "L", "R"}

COLOURS = {"Red": [255, 0, 0],
           "Green": [0, 255, 0],
           "Blue": [0, 0, 255],
           "Yellow": [255, 255, 0],
           "White": [255, 255, 255],
           "Orange": [255, 128, 0]}


START_CUBE = {
        # Red Side (Front)
        "F": {
            "TL": "Red", "TM": "Red", "TR": "Red",
            "ML": "Red", "MM": "Red", "MR": "Red",
            "BL": "Red", "BM": "Red", "BR": "Red"
            },

        # Green Side (Left)
        "L": {
            "TL": "Green", "TM": "Green", "TR": "Green",
            "ML": "Green", "MM": "Green", "MR": "Green",
            "BL": "Green", "BM": "Green", "BR": "Green"
            },

        # White Side (Up)
        "U": {
            "TL": "White", "TM": "White", "TR": "White",
            "ML": "White", "MM": "White", "MR": "White",
            "BL": "White", "BM": "White", "BR": "White"
            },

        # Blue Side (Right)
        "R": {
            "TL": "Blue", "TM": "Blue", "TR": "Blue",
            "ML": "Blue", "MM": "Blue", "MR": "Blue",
            "BL": "Blue", "BM": "Blue", "BR": "Blue"
            },

        # Yellow Side (Down)
        "D": {
            "TL": "Yellow", "TM": "Yellow", "TR": "Yellow",
            "ML": "Yellow", "MM": "Yellow", "MR": "Yellow",
            "BL": "Yellow", "BM": "Yellow", "BR": "Yellow"
            },

        # Orange Side (Back)
        "B": {
            "TL": "Orange", "TM": "Orange", "TR": "Orange",
            "ML": "Orange", "MM": "Orange", "MR": "Orange",
            "BL": "Orange", "BM": "Orange", "BR": "Orange"
            }
        }

CUBE_NEIGHBOURS = {
                    "F":
                    {
                        "U": ["U", "BL", "BM", "BR"],
                        "D": ["D", "TL", "TM", "TR"],
                        "L": ["L", "TR", "MR", "BR"],
                        "R": ["R", "TL", "ML", "BL"]
                    },
                    "B":
                    {
                        "U": ["U", "TR", "TM", "TL"],
                        "D": ["D", "BR", "BM", "BL"],
                        "L": ["R", "TR", "MR", "BR"],
                        "R": ["L", "TL", "ML", "BL"]
                    },
                    "U":
                    {
                        "U": ["B", "TR", "TM", "TL"],
                        "D": ["F", "TL", "TM", "TR"],
                        "L": ["L", "TL", "TM", "TR"],
                        "R": ["R", "TR", "TM", "TL"]
                    },
                    "D":
                    {
                        "U": ["F", "BL", "BM", "BR"],
                        "D": ["B", "BR", "BM", "BL"],
                        "L": ["L", "BR", "BM", "BL"],
                        "R": ["R", "BL", "BM", "BR"]
                    },
                    "L":
                    {
                        "U": ["U", "TL", "ML", "BL"],
                        "D": ["D", "BL", "ML", "TL"],
                        "L": ["B", "TR", "MR", "BR"],
                        "R": ["F", "TL", "ML", "BL"]
                    },
                    "R":
                    {
                        "U": ["U", "BR", "MR", "TR"],
                        "D": ["D", "TR", "MR", "BR"],
                        "L": ["F", "TR", "MR", "BR"],
                        "R": ["B", "TL", "ML", "BL"]
                    }
                    }


def generate_model(cube: dict):
    vertices = np.array([  # front face
                         [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
                         [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0],
                         [0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0],
                         [0, 3, 0], [1, 3, 0], [2, 3, 0], [3, 3, 0],
                         # back face
                         [0, 0, 3], [1, 0, 3], [2, 0, 3], [3, 0, 3],
                         [0, 1, 3], [1, 1, 3], [2, 1, 3], [3, 1, 3],
                         [0, 2, 3], [1, 2, 3], [2, 2, 3], [3, 2, 3],
                         [0, 3, 3], [1, 3, 3], [2, 3, 3], [3, 3, 3],
                         # left face
                         [0, 0, 1], [0, 0, 2],
                         [0, 1, 1], [0, 1, 2],
                         [0, 2, 1], [0, 2, 2],
                         [0, 3, 1], [0, 3, 2],
                         # right face
                         [3, 0, 1], [3, 0, 2],
                         [3, 1, 1], [3, 1, 2],
                         [3, 2, 1], [3, 2, 2],
                         [3, 3, 1], [3, 3, 2],
                         # up face
                         [1, 3, 1], [2, 3, 1],
                         [1, 3, 2], [2, 3, 2],
                         # down face
                         [1, 0, 1], [2, 0, 1],
                         [1, 0, 2], [2, 0, 2],
                         ])
    faces = np.hstack([  # Front
                       [4, 0, 1, 5, 4], [4, 1, 2, 6, 5], [4, 2, 3, 7, 6],  # R1
                       [4, 4, 5, 9, 8], [4, 5, 6, 10, 9], [4, 6, 7, 11, 10],  # R2
                       [4, 8, 9, 13, 12], [4, 9, 10, 14, 13], [4, 10, 11, 15, 14],  # R3
                       # Back
                       [4, 16, 17, 21, 20], [4, 17, 18, 22, 21], [4, 18, 19, 23, 22],  # R1
                       [4, 20, 21, 25, 24], [4, 21, 22, 26, 25], [4, 22, 23, 27, 26],  # R2
                       [4, 24, 25, 29, 28], [4, 25, 26, 30, 29], [4, 26, 27, 31, 30],  # R3
                       # Left
                       [4, 0, 32, 34, 4], [4, 32, 33, 35, 34], [4, 33, 16, 20, 35],  # R1
                       [4, 4, 34, 36, 8], [4, 34, 35, 37, 36], [4, 35, 20, 24, 37],  # R2
                       [4, 8, 36, 38, 12], [4, 36, 37, 39, 38], [4, 37, 24, 28, 39],  # R3
                       # Right
                       [4, 3, 40, 42, 7], [4, 40, 41, 43, 42], [4, 41, 19, 23, 43],  # R1
                       [4, 7, 42, 44, 11], [4, 42, 43, 45, 44], [4, 43, 23, 27, 45],  # R2
                       [4, 11, 44, 46, 15], [4, 44, 45, 47, 46], [4, 45, 27, 31, 47],  # R3
                       # Up
                       [4, 12, 13, 48, 38], [4, 13, 14, 49, 48], [4, 14, 15, 46, 49],  # R1
                       [4, 38, 48, 50, 39], [4, 48, 49, 51, 50], [4, 49, 46, 47, 51],  # R2
                       [4, 39, 50, 29, 28], [4, 50, 51, 30, 29], [4, 51, 47, 31, 30],  # R3
                       # Down
                       [4, 0, 1, 52, 32], [4, 1, 2, 53, 52], [4, 2, 3, 40, 53],  # R1
                       [4, 32, 52, 54, 33], [4, 52, 53, 55, 54], [4, 53, 40, 41, 55],  # R2
                       [4, 33, 54, 17, 16], [4, 54, 55, 18, 17], [4, 55, 41, 19, 18]  # R3
                       ])
    mesh = pv.PolyData(vertices, faces)
    mesh.cell_data['colors'] = [  # Front
                                COLOURS[cube["F"]["BL"]], COLOURS[cube["F"]["BM"]], COLOURS[cube["F"]["BR"]],  # R1
                                COLOURS[cube["F"]["ML"]], COLOURS[cube["F"]["MM"]], COLOURS[cube["F"]["MR"]],  # R2
                                COLOURS[cube["F"]["TL"]], COLOURS[cube["F"]["TM"]], COLOURS[cube["F"]["TR"]],  # R3
                                # Back
                                COLOURS[cube["B"]["BL"]], COLOURS[cube["B"]["BM"]], COLOURS[cube["B"]["BR"]],  # R1
                                COLOURS[cube["B"]["ML"]], COLOURS[cube["B"]["MM"]], COLOURS[cube["B"]["MR"]],  # R2
                                COLOURS[cube["B"]["TL"]], COLOURS[cube["B"]["TM"]], COLOURS[cube["B"]["TR"]],  # R3
                                # Left
                                COLOURS[cube["L"]["BL"]], COLOURS[cube["L"]["BM"]], COLOURS[cube["L"]["BR"]],  # R1
                                COLOURS[cube["L"]["ML"]], COLOURS[cube["L"]["MM"]], COLOURS[cube["L"]["MR"]],  # R2
                                COLOURS[cube["L"]["TL"]], COLOURS[cube["L"]["TM"]], COLOURS[cube["L"]["TR"]],  # R3
                                # Right
                                COLOURS[cube["R"]["BL"]], COLOURS[cube["R"]["BM"]], COLOURS[cube["R"]["BR"]],  # R1
                                COLOURS[cube["R"]["ML"]], COLOURS[cube["R"]["MM"]], COLOURS[cube["R"]["MR"]],  # R2
                                COLOURS[cube["R"]["TL"]], COLOURS[cube["R"]["TM"]], COLOURS[cube["R"]["TR"]],  # R3
                                # Up
                                COLOURS[cube["U"]["BL"]], COLOURS[cube["U"]["BM"]], COLOURS[cube["U"]["BR"]],  # R1
                                COLOURS[cube["U"]["ML"]], COLOURS[cube["U"]["MM"]], COLOURS[cube["U"]["MR"]],  # R2
                                COLOURS[cube["U"]["TL"]], COLOURS[cube["U"]["TM"]], COLOURS[cube["U"]["TR"]],  # R3
                                # Down
                                COLOURS[cube["D"]["BL"]], COLOURS[cube["D"]["BM"]], COLOURS[cube["D"]["BR"]],  # R1
                                COLOURS[cube["D"]["ML"]], COLOURS[cube["D"]["MM"]], COLOURS[cube["D"]["MR"]],  # R2
                                COLOURS[cube["D"]["TL"]], COLOURS[cube["D"]["TM"]], COLOURS[cube["D"]["TR"]]]  # R3
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='colors', lighting=False, rgb=True, preference='cell', show_edges=True)
    plotter.show()


def rotate_face(cube: dict, side: str) -> dict:
    # one rotation clockwise
    new_cube = copy.deepcopy(cube)
    neighbours_u = list(CUBE_NEIGHBOURS[side]["U"])
    neighbours_d = list(CUBE_NEIGHBOURS[side]["D"])
    neighbours_l = list(CUBE_NEIGHBOURS[side]["L"])
    neighbours_r = list(CUBE_NEIGHBOURS[side]["R"])
    # Face top row
    new_cube[side]["TL"] = cube[side]["BL"]
    new_cube[side]["TM"] = cube[side]["ML"]
    new_cube[side]["TR"] = cube[side]["TL"]
    # Face middle row
    new_cube[side]["ML"] = cube[side]["BM"]
    new_cube[side]["MM"] = cube[side]["MM"]
    new_cube[side]["MR"] = cube[side]["TM"]
    # Face bottom row
    new_cube[side]["BL"] = cube[side]["BR"]
    new_cube[side]["BM"] = cube[side]["MR"]
    new_cube[side]["BR"] = cube[side]["TR"]
    # Up
    new_cube[neighbours_u[0]][neighbours_u[1]] = cube[neighbours_l[0]][neighbours_l[3]]
    new_cube[neighbours_u[0]][neighbours_u[2]] = cube[neighbours_l[0]][neighbours_l[2]]
    new_cube[neighbours_u[0]][neighbours_u[3]] = cube[neighbours_l[0]][neighbours_l[1]]
    # Down
    new_cube[neighbours_d[0]][neighbours_d[1]] = cube[neighbours_r[0]][neighbours_r[3]]
    new_cube[neighbours_d[0]][neighbours_d[2]] = cube[neighbours_r[0]][neighbours_r[2]]
    new_cube[neighbours_d[0]][neighbours_d[3]] = cube[neighbours_r[0]][neighbours_r[1]]
    # Left
    new_cube[neighbours_l[0]][neighbours_l[1]] = cube[neighbours_d[0]][neighbours_d[1]]
    new_cube[neighbours_l[0]][neighbours_l[2]] = cube[neighbours_d[0]][neighbours_d[2]]
    new_cube[neighbours_l[0]][neighbours_l[3]] = cube[neighbours_d[0]][neighbours_d[3]]
    # Right
    new_cube[neighbours_r[0]][neighbours_r[1]] = cube[neighbours_u[0]][neighbours_u[1]]
    new_cube[neighbours_r[0]][neighbours_r[2]] = cube[neighbours_u[0]][neighbours_u[2]]
    new_cube[neighbours_r[0]][neighbours_r[3]] = cube[neighbours_u[0]][neighbours_u[3]]

    return new_cube


def print_cube(cube: dict):
    print(f'\n{cube["U"]["TL"]:<2} {cube["U"]["TM"]:<2} {cube["U"]["TR"]:<2}')
    print(f'{cube["U"]["ML"]:<2} {cube["U"]["MM"]:<2} {cube["U"]["MR"]:<2}')
    print(f'{cube["U"]["BL"]:<2} {cube["U"]["BM"]:<2} {cube["U"]["BR"]:<2}\n')
    print(f'{cube["F"]["TL"]:<2} {cube["F"]["TM"]:<2} {cube["F"]["TR"]:<2}\
    {cube["R"]["TL"]:<2} {cube["R"]["TM"]:<2} {cube["R"]["TR"]:<2}\
    {cube["B"]["TL"]:<2} {cube["B"]["TM"]:<2} {cube["B"]["TR"]:<2}\
    {cube["L"]["TL"]:<2} {cube["L"]["TM"]:<2} {cube["L"]["TR"]:<2}')
    print(f'{cube["F"]["ML"]:<2} {cube["F"]["MM"]:<2} {cube["F"]["MR"]:<2}\
    {cube["R"]["ML"]:<2} {cube["R"]["MM"]:<2} {cube["R"]["MR"]:<2}\
    {cube["B"]["ML"]:<2} {cube["B"]["MM"]:<2} {cube["B"]["MR"]:<2}\
    {cube["L"]["ML"]:<2} {cube["L"]["MM"]:<2} {cube["L"]["MR"]:<2}')
    print(f'{cube["F"]["BL"]:<2} {cube["F"]["BM"]:<2} {cube["F"]["BR"]:<2}\
    {cube["R"]["BL"]:<2} {cube["R"]["BM"]:<2} {cube["R"]["BR"]:<2}\
    {cube["B"]["BL"]:<2} {cube["B"]["BM"]:<2} {cube["B"]["BR"]:<2}\
    {cube["L"]["BL"]:<2} {cube["L"]["BM"]:<2} {cube["L"]["BR"]:<2}\n')
    print(f'{cube["D"]["TL"]:<2} {cube["D"]["TM"]:<2} {cube["D"]["TR"]:<2}')
    print(f'{cube["D"]["ML"]:<2} {cube["D"]["MM"]:<2} {cube["D"]["MR"]:<2}')
    print(f'{cube["D"]["BL"]:<2} {cube["D"]["BM"]:<2} {cube["D"]["BR"]:<2}')


def print_face(cube: dict, face: str):
    print(f'{cube[face]["TL"]:<2} {cube[face]["TM"]:<2} {cube[face]["TR"]:<2}')
    print(f'{cube[face]["ML"]:<2} {cube[face]["MM"]:<2} {cube[face]["MR"]:<2}')
    print(f'{cube[face]["BL"]:<2} {cube[face]["BM"]:<2} {cube[face]["BR"]:<2}')


def randomise_cube(cube: dict) -> dict:
    new_cube = copy.deepcopy(cube)
    moves_list = list(MOVES)
    print(moves_list)
    for _ in range(50):
        move = moves_list[random.randrange(0, 6)]
        print(move)
        new_cube = rotate_face(new_cube, move)
    return new_cube


def main():
    
    current_cube = copy.deepcopy(START_CUBE)
    print_cube(current_cube)
    generate_model(current_cube)
    current_cube = randomise_cube(current_cube)
    generate_model(current_cube)

    while START_CUBE != current_cube:
        user_choice = input(f"Choose a face to rotate (clockwise) must be one of: {MOVES}\n\
                            Choice: ").upper()
        if user_choice in MOVES:
            current_cube = rotate_face(current_cube, user_choice)
            generate_model(current_cube)


if __name__ == "__main__":
    main()
