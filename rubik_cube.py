import copy
import random
import sys

import pyvista as pv
from pyvistaqt import plotting
from PyQt6 import QtWidgets
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

CUBE_COORDINATES = {"F":
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0],
                     [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [3.0, 0.0, 2.0],
                     [0.0, 0.0, 3.0], [1.0, 0.0, 3.0], [2.0, 0.0, 3.0], [3.0, 0.0, 3.0]],
                    "B":
                    [[0.0, 3.0, 0.0], [1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0],
                     [0.0, 3.0, 1.0], [1.0, 3.0, 1.0], [2.0, 3.0, 1.0], [3.0, 3.0, 1.0],
                     [0.0, 3.0, 2.0], [1.0, 3.0, 2.0], [2.0, 3.0, 2.0], [3.0, 3.0, 2.0],
                     [0.0, 3.0, 3.0], [1.0, 3.0, 3.0], [2.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
                    "L":
                    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0],
                     [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 3.0, 1.0],
                     [0.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 2.0, 2.0], [0.0, 3.0, 2.0],
                     [0.0, 0.0, 3.0], [0.0, 1.0, 3.0], [0.0, 2.0, 3.0], [0.0, 3.0, 3.0]],
                    "R":
                    [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0],
                     [3.0, 0.0, 1.0], [3.0, 1.0, 1.0], [3.0, 2.0, 1.0], [3.0, 3.0, 1.0],
                     [3.0, 0.0, 2.0], [3.0, 1.0, 2.0], [3.0, 2.0, 2.0], [3.0, 3.0, 2.0],
                     [3.0, 0.0, 3.0], [3.0, 1.0, 3.0], [3.0, 2.0, 3.0], [3.0, 3.0, 3.0]],
                    "U":
                    [[0.0, 0.0, 3.0], [1.0, 0.0, 3.0], [2.0, 0.0, 3.0], [3.0, 0.0, 3.0],
                     [0.0, 1.0, 3.0], [1.0, 1.0, 3.0], [2.0, 1.0, 3.0], [3.0, 1.0, 3.0],
                     [0.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [3.0, 2.0, 3.0],
                     [0.0, 3.0, 3.0], [1.0, 3.0, 3.0], [2.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
                    "D":
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0],
                     [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0],
                     [0.0, 3.0, 0.0], [1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0]]}

VERTICES = np.array([
                    # Front
                    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0],
                    [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [3.0, 0.0, 2.0],
                    [0.0, 0.0, 3.0], [1.0, 0.0, 3.0], [2.0, 0.0, 3.0], [3.0, 0.0, 3.0],
                    # Back
                    [0.0, 3.0, 0.0], [1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0],
                    [0.0, 3.0, 1.0], [1.0, 3.0, 1.0], [2.0, 3.0, 1.0], [3.0, 3.0, 1.0],
                    [0.0, 3.0, 2.0], [1.0, 3.0, 2.0], [2.0, 3.0, 2.0], [3.0, 3.0, 2.0],
                    [0.0, 3.0, 3.0], [1.0, 3.0, 3.0], [2.0, 3.0, 3.0], [3.0, 3.0, 3.0],
                    # Left
                    [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0],
                    [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 2.0, 1.0], [0.0, 3.0, 1.0],
                    [0.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 2.0, 2.0], [0.0, 3.0, 2.0],
                    [0.0, 0.0, 3.0], [0.0, 1.0, 3.0], [0.0, 2.0, 3.0], [0.0, 3.0, 3.0],
                    # Right
                    [3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0], [3.0, 3.0, 0.0],
                    [3.0, 0.0, 1.0], [3.0, 1.0, 1.0], [3.0, 2.0, 1.0], [3.0, 3.0, 1.0],
                    [3.0, 0.0, 2.0], [3.0, 1.0, 2.0], [3.0, 2.0, 2.0], [3.0, 3.0, 2.0],
                    [3.0, 0.0, 3.0], [3.0, 1.0, 3.0], [3.0, 2.0, 3.0], [3.0, 3.0, 3.0],
                    # Up
                    [0.0, 0.0, 3.0], [1.0, 0.0, 3.0], [2.0, 0.0, 3.0], [3.0, 0.0, 3.0],
                    [0.0, 1.0, 3.0], [1.0, 1.0, 3.0], [2.0, 1.0, 3.0], [3.0, 1.0, 3.0],
                    [0.0, 2.0, 3.0], [1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [3.0, 2.0, 3.0],
                    [0.0, 3.0, 3.0], [1.0, 3.0, 3.0], [2.0, 3.0, 3.0], [3.0, 3.0, 3.0],
                    # Down
                    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0],
                    [0.0, 3.0, 0.0], [1.0, 3.0, 0.0], [2.0, 3.0, 0.0], [3.0, 3.0, 0.0]])

FACES = np.hstack([  # Front
                    [4, 0, 1, 5, 4], [4, 1, 2, 6, 5], [4, 2, 3, 7, 6],  # R1 L -> R
                    [4, 4, 5, 9, 8], [4, 5, 6, 10, 9], [4, 6, 7, 11, 10],  # R2 L -> R
                    [4, 8, 9, 13, 12], [4, 9, 10, 14, 13], [4, 10, 11, 15, 14],  # R3 L -> R
                    # Back
                    [4, 16, 17, 21, 20], [4, 17, 18, 22, 21], [4, 18, 19, 23, 22],  # R1
                    [4, 20, 21, 25, 24], [4, 21, 22, 26, 25], [4, 22, 23, 27, 26],  # R2
                    [4, 24, 25, 29, 28], [4, 25, 26, 30, 29], [4, 26, 27, 31, 30],  # R3
                    # Left
                    [4, 32, 33, 37, 36], [4, 33, 34, 38, 37], [4, 34, 35, 39, 38],  # R1
                    [4, 36, 37, 41, 40], [4, 37, 38, 42, 41], [4, 38, 39, 43, 42],  # R2
                    [4, 40, 41, 45, 44], [4, 41, 42, 46, 45], [4, 42, 43, 47, 46],  # R3
                    # Right
                    [4, 48, 49, 53, 52], [4, 49, 50, 54, 53], [4, 50, 51, 55, 54],  # R1
                    [4, 52, 53, 57, 56], [4, 53, 54, 58, 57], [4, 54, 55, 59, 58],  # R2
                    [4, 56, 57, 61, 60], [4, 57, 58, 62, 61], [4, 58, 59, 63, 62],  # R3
                    # Up
                    [4, 64, 65, 69, 68], [4, 65, 66, 70, 69], [4, 66, 67, 71, 70],  # R1
                    [4, 68, 69, 73, 72], [4, 69, 70, 74, 73], [4, 70, 71, 75, 74],  # R2
                    [4, 72, 73, 77, 76], [4, 73, 74, 78, 77], [4, 74, 75, 79, 78],  # R3
                    # Down
                    [4, 80, 81, 85, 84], [4, 81, 82, 86, 85], [4, 82, 83, 87, 86],  # R1
                    [4, 84, 85, 89, 88], [4, 85, 86, 90, 89], [4, 86, 87, 91, 90],  # R2
                    [4, 88, 89, 93, 92], [4, 89, 90, 94, 93], [4, 90, 91, 95, 94]   # R3
                    ])


# Initialise a global variable for the game cube based on a solved cube
current_cube = dict()


def generate_mesh(cube: dict) -> pv.PolyData:
    mesh = pv.PolyData(VERTICES, FACES)
    mesh.cell_data['colors'] = [
                                # Front
                                COLOURS[cube["F"]["BL"]], COLOURS[cube["F"]["BM"]], COLOURS[cube["F"]["BR"]],  # R1
                                COLOURS[cube["F"]["ML"]], COLOURS[cube["F"]["MM"]], COLOURS[cube["F"]["MR"]],  # R2
                                COLOURS[cube["F"]["TL"]], COLOURS[cube["F"]["TM"]], COLOURS[cube["F"]["TR"]],  # R3
                                # Back
                                COLOURS[cube["B"]["BR"]], COLOURS[cube["B"]["BM"]], COLOURS[cube["B"]["BL"]],  # R1
                                COLOURS[cube["B"]["MR"]], COLOURS[cube["B"]["MM"]], COLOURS[cube["B"]["ML"]],  # R2
                                COLOURS[cube["B"]["TR"]], COLOURS[cube["B"]["TM"]], COLOURS[cube["B"]["TL"]],  # R3
                                # Left
                                COLOURS[cube["L"]["BR"]], COLOURS[cube["L"]["BM"]], COLOURS[cube["L"]["BL"]],  # R1
                                COLOURS[cube["L"]["MR"]], COLOURS[cube["L"]["MM"]], COLOURS[cube["L"]["ML"]],  # R2
                                COLOURS[cube["L"]["TR"]], COLOURS[cube["L"]["TM"]], COLOURS[cube["L"]["TL"]],  # R3
                                # Right
                                COLOURS[cube["R"]["BL"]], COLOURS[cube["R"]["BM"]], COLOURS[cube["R"]["BR"]],  # R1
                                COLOURS[cube["R"]["ML"]], COLOURS[cube["R"]["MM"]], COLOURS[cube["R"]["MR"]],  # R2
                                COLOURS[cube["R"]["TL"]], COLOURS[cube["R"]["TM"]], COLOURS[cube["R"]["TR"]],  # R3
                                # Up
                                COLOURS[cube["U"]["BL"]], COLOURS[cube["U"]["BM"]], COLOURS[cube["U"]["BR"]],  # R1
                                COLOURS[cube["U"]["ML"]], COLOURS[cube["U"]["MM"]], COLOURS[cube["U"]["MR"]],  # R2
                                COLOURS[cube["U"]["TL"]], COLOURS[cube["U"]["TM"]], COLOURS[cube["U"]["TR"]],  # R3
                                # Down
                                COLOURS[cube["D"]["TL"]], COLOURS[cube["D"]["TM"]], COLOURS[cube["D"]["TR"]],  # R3
                                COLOURS[cube["D"]["ML"]], COLOURS[cube["D"]["MM"]], COLOURS[cube["D"]["MR"]],  # R2
                                COLOURS[cube["D"]["BL"]], COLOURS[cube["D"]["BM"]], COLOURS[cube["D"]["BR"]]]  # R1
    return mesh


def generate_model() -> plotting.QtInteractor:
    global current_cube
    plotter = plotting.QtInteractor(auto_update=True)
    plotter = update_mesh(plotter)
    return plotter


def rotate_face(side: str, plotter: plotting.QtInteractor, cube: dict):
    global current_cube
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
    current_cube = new_cube
    plotter = update_mesh(plotter)


def randomise_cube(plotter: plotting.QtInteractor):
    global current_cube
    current_cube = copy.deepcopy(START_CUBE)
    moves_list = list(MOVES)
    for _ in range(100):
        move = moves_list[random.randrange(0, 6)]
        rotate_face(move, plotter, current_cube)


def reset_cube(plotter: plotting.QtInteractor):
    global current_cube
    current_cube = copy.deepcopy(START_CUBE)
    plotter = update_mesh(plotter)


def update_mesh(plotter: plotting.QtInteractor) -> plotting.QtInteractor:
    global current_cube
    plotter.add_mesh(generate_mesh(current_cube),
                     scalars='colors',
                     lighting=False,
                     rgb=True,
                     preference='cell',
                     show_edges=True)
    return plotter


def initialise_window() -> QtWidgets.QWidget:
    window = QtWidgets.QWidget()
    plotter = generate_model()

    randomise_button = QtWidgets.QPushButton("Randomise Cube")
    reset_button = QtWidgets.QPushButton("Reset Cube")
    rotate_f = QtWidgets.QPushButton("Rotate Red")
    reverse_f = QtWidgets.QPushButton("Reverse Red")
    rotate_r = QtWidgets.QPushButton("Rotate Blue")
    reverse_r = QtWidgets.QPushButton("Reverse Blue")
    rotate_b = QtWidgets.QPushButton("Rotate Orange")
    reverse_b = QtWidgets.QPushButton("Reverse Orange")
    rotate_l = QtWidgets.QPushButton("Rotate Green")
    reverse_l = QtWidgets.QPushButton("Reverse Green")
    rotate_u = QtWidgets.QPushButton("Rotate White")
    reverse_u = QtWidgets.QPushButton("Reverse White")
    rotate_d = QtWidgets.QPushButton("Rotate Yellow")
    reverse_d = QtWidgets.QPushButton("Reverse Yellow")
    rotate_middle = QtWidgets.QPushButton("Rotate Middle")
    reverse_middle = QtWidgets.QPushButton("Reverse Middle")
    rotate_equator = QtWidgets.QPushButton("Rotate Equator")
    reverse_equator = QtWidgets.QPushButton("Reverse Equator")
    rotate_standing = QtWidgets.QPushButton("Rotate Standing")
    reverse_standing = QtWidgets.QPushButton("Reverse Standing")

    layout_buttons_left = QtWidgets.QVBoxLayout()
    layout_buttons_right = QtWidgets.QVBoxLayout()

    layout_buttons_left.addWidget(randomise_button)
    layout_buttons_right.addWidget(reset_button)
    layout_buttons_left.addWidget(rotate_f)
    layout_buttons_right.addWidget(reverse_f)
    layout_buttons_left.addWidget(rotate_r)
    layout_buttons_right.addWidget(reverse_r)
    layout_buttons_left.addWidget(rotate_b)
    layout_buttons_right.addWidget(reverse_b)
    layout_buttons_left.addWidget(rotate_l)
    layout_buttons_right.addWidget(reverse_l)
    layout_buttons_left.addWidget(rotate_u)
    layout_buttons_right.addWidget(reverse_u)
    layout_buttons_left.addWidget(rotate_d)
    layout_buttons_right.addWidget(reverse_d)
    layout_buttons_left.addWidget(rotate_middle)
    layout_buttons_right.addWidget(reverse_middle)
    layout_buttons_left.addWidget(rotate_equator)
    layout_buttons_right.addWidget(reverse_equator)
    layout_buttons_left.addWidget(rotate_standing)
    layout_buttons_right.addWidget(reverse_standing)

    layout_window = QtWidgets.QHBoxLayout()
    layout_window.addWidget(plotter)
    layout_window.addLayout(layout_buttons_left)
    layout_window.addLayout(layout_buttons_right)

    window.setLayout(layout_window)
    window.setWindowTitle("Rubik's Cube")

    # Define actions for each button press
    randomise_button.clicked.connect(lambda: randomise_cube(plotter=plotter))
    rotate_f.clicked.connect(lambda: rotate_face("F", plotter, current_cube))
    reverse_f.clicked.connect(lambda: [rotate_face("F", plotter, current_cube) for _ in range(3)])
    rotate_r.clicked.connect(lambda: rotate_face("R", plotter, current_cube))
    reverse_r.clicked.connect(lambda: [rotate_face("R", plotter, current_cube) for _ in range(3)])
    rotate_b.clicked.connect(lambda: rotate_face("B", plotter, current_cube))
    reverse_b.clicked.connect(lambda: [rotate_face("B", plotter, current_cube) for _ in range(3)])
    rotate_l.clicked.connect(lambda: rotate_face("L", plotter, current_cube))
    reverse_l.clicked.connect(lambda: [rotate_face("L", plotter, current_cube) for _ in range(3)])
    rotate_u.clicked.connect(lambda: rotate_face("U", plotter, current_cube))
    reverse_u.clicked.connect(lambda: [rotate_face("U", plotter, current_cube) for _ in range(3)])
    rotate_d.clicked.connect(lambda: rotate_face("D", plotter, current_cube))
    reverse_d.clicked.connect(lambda: [rotate_face("D", plotter, current_cube) for _ in range(3)])
    rotate_middle.clicked.connect(lambda: [rotate_face(x, plotter, current_cube) for x in ["L", "R", "R", "R"]])
    reverse_middle.clicked.connect(lambda: [rotate_face(x, plotter, current_cube) for x in ["R", "L", "L", "L"]])
    rotate_equator.clicked.connect(lambda: [rotate_face(x, plotter, current_cube) for x in ["U", "U", "U", "D"]])
    reverse_equator.clicked.connect(lambda: [rotate_face(x, plotter, current_cube) for x in ["D", "D", "D", "U"]])
    rotate_standing.clicked.connect(lambda: [rotate_face(x, plotter, current_cube) for x in ["F", "F", "F", "B"]])
    reverse_standing.clicked.connect(lambda: [rotate_face(x, plotter, current_cube) for x in ["F", "B", "B", "B"]])
    reset_button.clicked.connect(lambda: reset_cube(plotter))
    return window


def main():
    global current_cube
    # Generate the cube to be used in game
    current_cube = copy.deepcopy(START_CUBE)
    # Initialise an app to display the cube
    app = QtWidgets.QApplication(sys.argv)

    # Initialise and show the game window
    window = initialise_window()
    window.show()

    app.exec_()


if __name__ == "__main__":
    main()
