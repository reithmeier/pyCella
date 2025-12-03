"""
hp
"""

import time

import cv2

from pycella import HostPathogen, CvGridWindow, CvLinePlotWindow, gaussian_matrix


def host_pathogen():
    grid_window = CvGridWindow(
        "HostPathogen",
        color_map={
            HostPathogen.states["empty"]: (222, 43, 22),  # blue
            HostPathogen.states["healthy"]: (34, 139, 34),  # green
            HostPathogen.states["infected"]: (0, 0, 255),  # red
        },
    )
    line_window = CvLinePlotWindow("Statistics")
    experiment = HostPathogen(
        50,
        50,
        0.1,
        0.8,
        boundary="wrap",
    )

    while experiment.has_next():
        experiment.next()
        grid_window.show(experiment.grid)
        line_window.show(
            [
                (experiment.stats_empty, (222, 43, 22)),
                (experiment.stats_healthy, (34, 139, 34)),
                (experiment.stats_infected, (0, 0, 255)),
            ]
        )

        # wait for ESC
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
            break

    # wait for key
    cv2.waitKey(0)


def main():
    host_pathogen()


if __name__ == "__main__":
    main()
