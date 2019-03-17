from math import *

# Based on original implementation by @elovenkova

class Point():
    def __init__(self, x, y, vx, vy, ra, la):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ra = ra    # if the contact with right neighbour exists (ra == 1) or is broken (ra == 0)
        self.la = la    # if the contact with left neighbour exists (ra == 1) or is broken (ra == 0)

    def move(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt


class Fiber():
    def __init__(self, LENGTH, DIAMETER, YOUNG_MODULUS, DENSITY, SPACE_STEP, PRESSURE_T, PRESSURE_R, PRESSURE_AMPLITUDE, DEFORMATION_LIMIT, BE_VERBOSE):
        self.verbose = BE_VERBOSE
        if self.verbose:
            print ("Creating fiber: %f %f %f %f %f" % (LENGTH, DIAMETER, YOUNG_MODULUS, DENSITY, DEFORMATION_LIMIT))
            print ("Space step: %f" % SPACE_STEP)
            print ("Pressure pulse params: %f %f %f" % (PRESSURE_T, PRESSURE_R, PRESSURE_AMPLITUDE))

        self.time_step_number = 0

        # store these fiber params as passed
        self.H = SPACE_STEP
        self.DIAMETER = DIAMETER
        self.E = YOUNG_MODULUS
        self.DENSITY = DENSITY
        self.PRESSURE_T = PRESSURE_T
        self.PRESSURE_R = PRESSURE_R
        self.PRESSURE_AMPLITUDE = PRESSURE_AMPLITUDE
        self.DEFORMATION_LIMIT = DEFORMATION_LIMIT

        # calculate ans store these internal params from passed values
        self.FIBER_SEGMENT_MASS = DENSITY*(0.25*pi*(DIAMETER**2)*SPACE_STEP)
        self.POINTS_NUMBER = int(LENGTH / SPACE_STEP) + 1

        self.T = 0
        self.points = []
        self.points_new = []

        X_0 = 0
        Y_0 = 0

        self.points.append(Point(X_0, Y_0, 0, 0, 1, 0))
        self.points_new.append(Point(X_0, Y_0, 0, 0, 1, 0))
        for i in range(1, self.POINTS_NUMBER - 1):
            self.points.append(Point(X_0 + SPACE_STEP * i, Y_0, 0, 0, 1, 1))
            self.points_new.append(Point(X_0 + SPACE_STEP * i, Y_0, 0, 0, 1, 1))
        self.points.append(Point(X_0 + SPACE_STEP * (self.POINTS_NUMBER - 1), Y_0, 0, 0, 0, 1))
        self.points_new.append(Point(X_0 + SPACE_STEP * (self.POINTS_NUMBER - 1), Y_0, 0, 0, 0, 1))

    def is_broken(self):
        for i in range(1, self.POINTS_NUMBER - 1):
            if self.points[i].ra == 0 or self.points[i].la == 0:
                return True
        return False

    def apply_pressure(self, n, dt):
        if self.T >= self.PRESSURE_T:
            return 0
        dist = fabs(self.points[n].x - self.points[self.POINTS_NUMBER//2].x)
        if dist <= self.PRESSURE_R:
            return ((self.PRESSURE_AMPLITUDE * self.DIAMETER * self.H)/self.FIBER_SEGMENT_MASS) * dt * cos((dist / self.PRESSURE_R) * pi / 2) ** 2
        else:
            return 0

    def step(self):
        # determine correct time step
        p_wave_speed = (self.E / self.DENSITY) ** 0.5
        self.dt = ((self.points[0].x - self.points[1].x)**2 + (self.points[0].y - self.points[1].y)**2) ** 0.5 / p_wave_speed
        for i in range(1, self.POINTS_NUMBER):
            if self.points[i].la == 0:
                continue
            tau = ((self.points[i-1].x - self.points[i].x)**2 + (self.points[i-1].y - self.points[i].y)**2) ** 0.5 / p_wave_speed
            self.dt = min(self.dt, tau)

        # handle fiber movement
        for i in range(0, self.POINTS_NUMBER):
            self.points[i].move(self.dt)

        # apply external pressure
        for i in range(0, self.POINTS_NUMBER):
            self.points_new[i].vy += self.apply_pressure(i, self.dt)

        # handle elastic forces
        for i in range(0, self.POINTS_NUMBER):
            eps_1 = eps_2 = 0
            s_1 = s_2 = c_1 = c_2 = 0

            # calculate force from left neighbour
            if i >= 1:
                d_1 = ((self.points[i-1].x - self.points[i].x)**2 + (self.points[i-1].y - self.points[i].y)**2)**0.5
                eps_1 = d_1 - self.H
                s_1 = (self.points[i-1].y - self.points[i].y)/d_1
                c_1 = (self.points[i-1].x - self.points[i].x)/d_1
            # calculate force from right neighbour
            if i < self.POINTS_NUMBER - 1:
                d_2 = ((self.points[i+1].x - self.points[i].x)**2 + (self.points[i+1].y - self.points[i].y)**2)**0.5
                eps_2 = d_2 - self.H
                s_2 = (self.points[i+1].y - self.points[i].y)/d_2
                c_2 = (self.points[i+1].x - self.points[i].x)/d_2

            # handle possible fiber fracture
            if eps_1 >= self.DEFORMATION_LIMIT * self.H:
                self.points_new[i].la = 0
            if eps_2 >= self.DEFORMATION_LIMIT*self.H:
                self.points_new[i].ra = 0

            # apply forces taking possible fracture into account
            if self.points[i].ra == 0 and self.points[i].la == 1:
                self.points_new[i].vy += s_1 * eps_1 * self.E * self.dt * self.DIAMETER ** 2 * pi / (4 * self.FIBER_SEGMENT_MASS * d_1)
                self.points_new[i].vx += c_1 * eps_1 * self.E * self.dt * self.DIAMETER ** 2 * pi / (4 * self.FIBER_SEGMENT_MASS * d_1)
            elif self.points[i].la == 0 and self.points[i].ra == 1:
                self.points_new[i].vy += s_2 * eps_2 * self.E * self.dt * self.DIAMETER ** 2 * pi / (4 * self.FIBER_SEGMENT_MASS * d_2)
                self.points_new[i].vx += c_2 * eps_2 * self.E * self.dt * self.DIAMETER ** 2 * pi / (4 * self.FIBER_SEGMENT_MASS * d_2)
            elif self.points[i].la == 1 and self.points[i].ra == 1:
                self.points_new[i].vy += (s_1*eps_1/d_1 + s_2*eps_2/d_2) * self.E * self.dt * self.DIAMETER ** 2 * pi / (4 * self.FIBER_SEGMENT_MASS)
                self.points_new[i].vx += (c_1*eps_1/d_1 + c_2*eps_2/d_2) * self.E * self.dt * self.DIAMETER ** 2 * pi / (4 * self.FIBER_SEGMENT_MASS)

        # advance to the next time step
        for i in range(0, self.POINTS_NUMBER):
            self.points[i].vx = self.points_new[i].vx
            self.points[i].vy = self.points_new[i].vy
            self.points[i].ra = self.points_new[i].ra
            self.points[i].la = self.points_new[i].la
        self.time_step_number += 1
        self.T += self.dt
        if self.verbose:
            print("Done step %d with time step %.9f, total time is %.9f" % (self.time_step_number, self.dt, self.T))
