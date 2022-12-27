import copy
import random


start_cube = {
        # Red Side (Front)
        "F": {
            "TL": "R", "TM": "R", "TR": "R",
            "ML": "R", "MM": "R", "MR": "R",
            "BL": "R", "BM": "R", "BR": "R"
            },

        # Green Side (Left)
        "L": {
            "TL": "G", "TM": "G", "TR": "G",
            "ML": "G", "MM": "G", "MR": "G",
            "BL": "G", "BM": "G", "BR": "G"
            },

        # White Side (Up)
        "U": {
            "TL": "W", "TM": "W", "TR": "W",
            "ML": "W", "MM": "W", "MR": "W",
            "BL": "W", "BM": "W", "BR": "W"
            },

        # Blue Side (Right)
        "R": {
            "TL": "B", "TM": "B", "TR": "B",
            "ML": "B", "MM": "B", "MR": "B",
            "BL": "B", "BM": "B", "BR": "B"
            },

        # Yellow Side (Down)
        "D": {
            "TL": "Y", "TM": "Y", "TR": "Y",
            "ML": "Y", "MM": "Y", "MR": "Y",
            "BL": "Y", "BM": "Y", "BR": "Y"
            },

        # Orange Side (Back)
        "B": {
            "TL": "O", "TM": "O", "TR": "O",
            "ML": "O", "MM": "O", "MR": "O",
            "BL": "O", "BM": "O", "BR": "O"
            }
        }

cube_neighbours = {
                    "F":
                    {
                        "U": ["U", "BL", "BM", "BR"],
                        "D": ["D", "TL", "TM", "TR"],
                        "L": ["L", "TR", "MR", "BR"],
                        "R": ["R", "TL", "ML", "BL"]
                    },
                    "B":
                    {
                        "U": ["U", "TL", "TM", "TR"],
                        "D": ["D", "BL", "BM", "BR"],
                        "L": ["R", "TR", "MR", "BR"],
                        "R": ["L", "TL", "ML", "BL"]
                    },
                    "U":
                    {
                        "U": ["B", "TL", "TM", "TR"],
                        "D": ["F", "TL", "TM", "TR"],
                        "L": ["L", "TL", "TM", "TR"],
                        "R": ["R", "TL", "TM", "TR"]
                    },
                    "D":
                    {
                        "U": ["F", "BL", "BM", "BR"],
                        "D": ["B", "BL", "BM", "BR"],
                        "L": ["L", "BL", "BM", "BR"],
                        "R": ["R", "BL", "BM", "BR"]
                    },
                    "L":
                    {
                        "U": ["U", "TL", "ML", "BL"],
                        "D": ["D", "TL", "ML", "BL"],
                        "L": ["B", "TR", "MR", "BR"],
                        "R": ["F", "TL", "ML", "BL"]
                    },
                    "R":
                    {
                        "U": ["U", "TR", "MR", "BR"],
                        "D": ["D", "TR", "MR", "BR"],
                        "L": ["F", "TR", "MR", "BR"],
                        "R": ["B", "TL", "ML", "BL"]
                    }
                    }


MOVES = {"U", "D", "F", "B", "L", "R"}


def rotate_face(cube: dict, side: str) -> dict:
    # one rotation clockwise
    new_cube = copy.deepcopy(cube)
    neighbours_u = list(cube_neighbours[side]["U"])
    neighbours_d = list(cube_neighbours[side]["D"])
    neighbours_l = list(cube_neighbours[side]["L"])
    neighbours_r = list(cube_neighbours[side]["R"])
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


def print_cube(cube):
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


def randomise_cube(cube: dict):
    new_cube = copy.deepcopy(cube)
    moves_list = list(MOVES)
    print(moves_list)
    for _ in range(50):
        move = moves_list[random.randrange(0, 6)]
        print(move)
        new_cube = rotate_face(new_cube, move)
    return new_cube


def main():
    current_cube = start_cube.copy()
    print_cube(current_cube)

    current_cube = randomise_cube(current_cube)
    print_cube(current_cube)

    for _ in range(10):
        user_choice = input(f"Choose a face to rotate (clockwise) must be one of: {MOVES}\n\
                            Choice: ").upper()
        if user_choice in MOVES:
            current_cube = rotate_face(current_cube, user_choice)
            print_cube(current_cube)


if __name__ == "__main__":
    main()
