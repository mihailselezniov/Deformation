#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# These are for visualization only
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# This is for handling command line arguments
import argparse

# This is our fiber, import it from separate file
from fiber import *


# These two are pure visualization functions for FuncAnimate
# If you do not need visualization, you do not need these functions

def init_frame():
    line.set_data([], [])
    return line,


def next_frame(frame_number, fiber):
    #print("Frame number %d" % frame_number)
    x = []
    y = []
    fiber.step()
    for i in range(0, fiber.POINTS_NUMBER):
        if fiber.points[i].la == 1 and fiber.points[i].ra == 1:
            x.append(fiber.points[i].x)
            y.append(fiber.points[i].y)
        if fiber.points[i].la == 0 and fiber.points[i].ra == 1:
            x.append(np.NaN)
            y.append(np.NaN)
            x.append(fiber.points[i].x)
            y.append(fiber.points[i].y)
        if fiber.points[i].ra == 0 and fiber.points[i].la == 1:
            x.append(fiber.points[i].x)
            y.append(fiber.points[i].y)
            x.append(np.NaN)
            y.append(np.NaN)
        if fiber.points[i].ra == 0 and fiber.points[i].la == 0:
            # Kind of damned magic happens here, this part can be done better
            x.append(np.NaN)
            y.append(np.NaN)
            dx1 = dx2 = dy1 = dy2 = 0
            d1 = d2 = 1
            if i > 0:
                dx1 = fiber.points[i-1].x - fiber.points[i].x
                dy1 = fiber.points[i-1].y - fiber.points[i].y
                d1 = (dx1 ** 2 + dy1 ** 2) ** 0.5
            if i < fiber.POINTS_NUMBER - 1:
                dx2 = fiber.points[i+1].x - fiber.points[i].x
                dy2 = fiber.points[i+1].y - fiber.points[i].y
                d2 = (dx2 ** 2 + dy2 ** 2) ** 0.5
            x.append(fiber.points[i].x + 0.1 * SPACE_STEP * dx1 / d1)
            y.append(fiber.points[i].y + 0.1 * SPACE_STEP * dy1 / d1)
            x.append(fiber.points[i].x + 0.1 * SPACE_STEP * dx2 / d2)
            y.append(fiber.points[i].y + 0.1 * SPACE_STEP * dy2 / d2)
            x.append(np.NaN)
            y.append(np.NaN)
    line.set_data(x, y)
    return line,


parser = argparse.ArgumentParser()
parser.add_argument('--verbose', help='be verbose instead of silent', type=bool, default=False)
parser.add_argument('--show', help='show process visually', type=bool, default=False)
parser.add_argument('--length', help='fiber length, in centimeters', type=float, default=10)
parser.add_argument('--diameter', help='fiber diameter, in millimeters', type=float, default=0.1)
parser.add_argument('--young', help='fiber young modulus, in GPa', type=float, default=125)
parser.add_argument('--density', help='fiber density, in kg/m3', type=float, default=1430)
parser.add_argument('--strength', help='fiber deformation limit, in percents', type=float, default=4)
parser.add_argument('--pressure_time', help='pressure pulse total time, in microseconds', type=float, default=10)
parser.add_argument('--pressure_radius', help='pressure pulse radius, in centimeters', type=float, default=1)
parser.add_argument('--pressure_amplitude', help='pressure pulse amplitude, in MPa', type=float, default=25)
args = parser.parse_args()

# numerical solver params
POINTS_PER_FIBER = 200
NUMBER_OF_FRAMES = 3000
SPACE_STEP = (args.length / 100) / POINTS_PER_FIBER

# Create fiber
fiber = Fiber(args.length / 100, args.diameter / 1000, args.young * 10**9, args.density, SPACE_STEP,
              args.pressure_time / 10**6, args.pressure_radius / 100, args.pressure_amplitude * 10**6, args.strength / 100,
              args.verbose)

if args.show:
    # Calc and show
    fig = plt.figure()
    ax = plt.axes(xlim=(-0.01, 0.11), ylim=(-0.01, 0.2001))
    line, = ax.plot([], [])
    anim = FuncAnimation(fig, next_frame, init_func=init_frame, repeat=False, frames=NUMBER_OF_FRAMES, interval=10,
                         blit=True, fargs={fiber})
    plt.show()
else:
    # Calc and check
    for frame_number in range(0, NUMBER_OF_FRAMES):
        fiber.step()

    if fiber.is_broken():
        print("Fiber is broken!")
    else:
        print("Fiber is intact!")
