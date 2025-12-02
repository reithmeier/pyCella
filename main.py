import cv2

from pycella import DrosselSchwabl, GameOfLife, gaussian_matrix, CvGridWindow, CvHistogramWindow


def drossel_schwabl():
    grid_window = CvGridWindow("DrosselSchwabl", color_map={
        DrosselSchwabl.states["free"]: (0, 0, 0),  # black
        DrosselSchwabl.states["tree"]: (34, 139, 34),  # green
        DrosselSchwabl.states["fire"]: (0, 0, 255)  # red
    })
    hist_window = CvHistogramWindow("FireStats", bins=64)
    experiment = DrosselSchwabl(50, 50, 0.001, 0.0001, neighborhood=gaussian_matrix(3, 3, 1.0), boundary='wrap')

    while True:
        experiment.next()
        grid_window.show(experiment.grid)
        hist_window.show(experiment.fire_stats)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
            break


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
    drossel_schwabl()
    game_of_life()


if __name__ == "__main__":
    main()
