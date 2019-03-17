#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fiber import *


class Test_drive_fiber():
    def __init__(self, fiber_params):
        self.fiber_params = fiber_params
        for key, value in fiber_params.items():
            self.__dict__[key] = value
        
        self.SPACE_STEP = (self.length / 100) / self.POINTS_PER_FIBER
        self.fiber = Fiber(self.length / 100,
                           self.diameter / 1000,
                           self.young * 10**9,
                           self.density,
                           self.SPACE_STEP,
                           self.pressure_time / 10**6,
                           self.pressure_radius / 100,
                           self.pressure_amplitude * 10**6,
                           self.strength / 100,
                           False)

    def is_broken(self):
        for frame_number in range(0, self.NUMBER_OF_FRAMES):
            self.fiber.step()
            if (not frame_number % 100) and self.fiber.is_broken():
                return True
        return self.fiber.is_broken()
    
    def get_results(self):
        self.fiber_params['is_broken'] = 1 if self.is_broken() else 0
        return self.fiber_params
        
