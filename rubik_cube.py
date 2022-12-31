import copy
import random

import pyvista as pv
from pyvistaqt import plotting
from PyQt6 import QtWidgets
import sys
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


def generate_mesh(cube: dict):
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


def generate_model(cube: dict, window: QtWidgets.QWidget) -> plotting.QtInteractor:
    plotter = plotting.QtInteractor(parent=window, auto_update=True)
    plotter = update_mesh(plotter, cube)
    return plotter


def rotate_face(cube: dict, side: str, plotter: plotting.QtInteractor) -> dict:
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
    plotter = update_mesh(plotter, cube)
    return new_cube


def randomise_cube(cube: dict, plotter: plotting.QtInteractor) -> dict:
    new_cube = copy.deepcopy(cube)
    moves_list = list(MOVES)
    for _ in range(50):
        move = moves_list[random.randrange(0, 6)]
        new_cube = rotate_face(new_cube, move, plotter)
    return new_cube


def update_mesh(plotter: plotting.QtInteractor, current_cube: dict) -> plotting.QtInteractor:
    plotter.add_mesh(generate_mesh(current_cube),
                     scalars='colors',
                     lighting=False,
                     rgb=True,
                     preference='cell',
                     show_edges=True)
    return plotter


def main():
    # Initialise an app to display the cube
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QWidget()
    # Initialise a game cube based on a solved cube
    current_cube = copy.deepcopy(START_CUBE)
    # Generage a model of the cube to display
    plotter = generate_model(current_cube, window)
    # Display the window
    window.show()
    
    button1 = QtWidgets.QPushButton("One")
    button2 = QtWidgets.QPushButton("Two")
    button3 = QtWidgets.QPushButton("Three")
    button4 = QtWidgets.QPushButton("Four")
    button5 = QtWidgets.QPushButton("Five")

    layout = QtWidgets.QHBoxLayout()
    layout.addWidget(button1)
    layout.addWidget(button2)
    layout.addWidget(button3)
    layout.addWidget(button4)
    layout.addWidget(button5)

    window.setLayout(layout)
    # layout = QtWidgets.QHBoxLayout()

    randomise_cube_choice = input("Would you like to randomise the cube? (Y/N): ").upper()
    if randomise_cube_choice == "Y":
        current_cube = randomise_cube(current_cube, plotter)
        plotter = update_mesh(plotter, current_cube)

    while True:
        user_choice = input(f"Choose a face to rotate (clockwise) must be one of: {MOVES}\n\
                            Choice: ").upper()
        if user_choice in MOVES:
            current_cube = rotate_face(current_cube, user_choice, plotter)
            plotter = update_mesh(plotter, current_cube)


if __name__ == "__main__":
    main()
