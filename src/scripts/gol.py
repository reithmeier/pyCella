"""
gol
"""
import cv2

from pycella import GameOfLife, CvGridWindow


def game_of_life():
    experiment = GameOfLife(500, 500)
    grid_window = CvGridWindow("GameOfLife", color_map={
        GameOfLife.states["dead"]: (0, 0, 0),  # black
        GameOfLife.states["alive"]: (255, 255, 255),  # white
    })

    while True:
        experiment.next()
        grid_window.show(experiment.grid)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
            break


def main():
    game_of_life()


if __name__ == "__main__":
    main()
