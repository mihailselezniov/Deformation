#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fiber import *


length = 10
diameter = 0.1
young = 125
density = 1430
strength = 4
pressure_time = 10
pressure_radius = 1
pressure_amplitude = 25

# numerical solver params
POINTS_PER_FIBER = 200
NUMBER_OF_FRAMES = 3000
SPACE_STEP = (length / 100) / POINTS_PER_FIBER

fiber = Fiber(length/100, diameter/1000, young*10**9, density, SPACE_STEP, pressure_time/10**6,
              pressure_radius/100, pressure_amplitude*10**6, strength/100, False)

def is_broken_fiber(fiber):
    for frame_number in range(0, NUMBER_OF_FRAMES):
        fiber.step()
        if (not frame_number%100) and fiber.is_broken():
            return True
    return fiber.is_broken()

print(is_broken_fiber(fiber))
