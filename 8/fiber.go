package main

import (
    "fmt"
    "math"
    "time"
)

type Point struct {
    x, y, vx, vy, ra, la float64
}

func (p *Point) move(dt float64) {
    p.x += p.vx * dt
    p.y += p.vy * dt
}

type Fiber struct {
    LENGTH, DIAMETER, E, DENSITY, H, PRESSURE_T, PRESSURE_R, PRESSURE_AMPLITUDE, DEFORMATION_LIMIT float64
    time_step_number, FIBER_SEGMENT_MASS, T, X_0, Y_0, dt, p_wave_speed float64
    POINTS_NUMBER, POINTS_PER_FIBER, NUMBER_OF_FRAMES, broken int
    points, points_new []Point
}

func init_fiber(length, diameter, young, density, pressure_time, pressure_radius, pressure_amplitude, strength float64) *Fiber {
    f := new(Fiber)

    // numerical solver params
    f.POINTS_PER_FIBER, f.NUMBER_OF_FRAMES = 200, 3000

    f.LENGTH, f.DIAMETER, f.E, f.DENSITY, f.H, f.PRESSURE_T, f.PRESSURE_R, f.PRESSURE_AMPLITUDE, f.DEFORMATION_LIMIT =
      length / 100.0, // fiber length, in centimeters
      diameter / 1000.0, // fiber diameter, in millimeters
      young * math.Pow(10, 9), // fiber young modulus, in GPa
      density, // fiber density, in kg/m3
      (length / 100.0) / float64(f.POINTS_PER_FIBER), // space step
      pressure_time / math.Pow(10, 6), // pressure pulse total time, in microseconds
      pressure_radius / 100.0, // pressure pulse radius, in centimeters
      pressure_amplitude * math.Pow(10, 6), // pressure pulse amplitude, in MPa
      strength / 100.0 // fiber deformation limit, in percents

    // calculate ans store these internal params from passed values
    f.FIBER_SEGMENT_MASS = f.DENSITY * (0.25 * math.Pi * math.Pow(f.DIAMETER, 2) * f.H)
    f.POINTS_NUMBER = int(f.LENGTH / f.H) + 1

    f.points = append(f.points, Point{f.X_0, f.Y_0, 0.0, 0.0, 1.0, 0.0})
    for i := 1; i < f.POINTS_NUMBER - 1; i++ {
        f.points = append(f.points, Point{f.X_0 + f.H * float64(i), f.Y_0, 0.0, 0.0, 1.0, 1.0})
    }
    f.points = append(f.points, Point{f.X_0 + f.H * float64(f.POINTS_NUMBER - 1), f.Y_0, 0.0, 0.0, 0.0, 1.0})
    f.points_new = make([]Point, len(f.points))
    copy(f.points_new, f.points)
    return f
}

func (f *Fiber) get_tau(i int) float64 {
    return math.Pow((math.Pow((f.points[i-1].x - f.points[i].x), 2) + math.Pow((f.points[i-1].y - f.points[i].y), 2)), 0.5) / f.p_wave_speed
}

func (f *Fiber) determine_correct_time_step() {
    f.p_wave_speed = math.Pow((f.E / f.DENSITY), 0.5)
    f.dt = f.get_tau(1)
    for i := 1; i < f.POINTS_NUMBER; i++ {
        if f.points[i].la == 0 {continue}
        f.dt = math.Min(f.dt, f.get_tau(i))
    }
}

func (f *Fiber) handle_fiber_movement() {
    for i := 0; i < f.POINTS_NUMBER; i++ {
        f.points[i].move(f.dt)
    }
}

func (f *Fiber) apply_pressure(n int) float64 {
    if f.T >= f.PRESSURE_T {return 0.0}
    dist := math.Abs(f.points[n].x - f.points[f.POINTS_NUMBER / 2].x)
    if dist <= f.PRESSURE_R {
        return ((f.PRESSURE_AMPLITUDE * f.DIAMETER * f.H) / f.FIBER_SEGMENT_MASS) * f.dt * math.Pow(math.Cos((dist / f.PRESSURE_R) * math.Pi / 2.0), 2)
    }
    return 0.0
}

func (f *Fiber) apply_external_pressure() {
    for i := 0; i < f.POINTS_NUMBER; i++ {
        f.points_new[i].vy += f.apply_pressure(i)
    }
}

func (f *Fiber) calculate_force_from_neighbour(i_neighbour, i int, e *Elastic) {
    e.d = math.Pow((math.Pow((f.points[i_neighbour].x - f.points[i].x), 2) + math.Pow((f.points[i_neighbour].y - f.points[i].y), 2)), 0.5)
    e.eps = e.d - f.H
    e.s = (f.points[i_neighbour].y - f.points[i].y) / e.d
    e.c = (f.points[i_neighbour].x - f.points[i].x) / e.d
}

type Elastic struct {
    d, eps, s, c float64
}

func (f *Fiber) handle_elastic_forcese() {
    for i := 0; i < f.POINTS_NUMBER; i++ {
        var e1, e2 Elastic

        // calculate force from left and right neighbour
        if i >= 1 {f.calculate_force_from_neighbour(i-1, i, &e1)}
        if i < f.POINTS_NUMBER - 1 {f.calculate_force_from_neighbour(i+1, i, &e2)}

        // handle possible fiber fracture
        if e1.eps >= f.DEFORMATION_LIMIT * f.H {f.points_new[i].la, f.broken = 0.0, 1}
        if e2.eps >= f.DEFORMATION_LIMIT * f.H {f.points_new[i].ra, f.broken = 0.0, 1}

        // apply forces taking possible fracture into account
        if (f.points[i].ra == 0) && (f.points[i].la == 1) {
            f.points_new[i].vy += e1.s * e1.eps * f.E * f.dt * math.Pow(f.DIAMETER, 2) * math.Pi / (4 * f.FIBER_SEGMENT_MASS * e1.d)
            f.points_new[i].vx += e1.c * e1.eps * f.E * f.dt * math.Pow(f.DIAMETER, 2) * math.Pi / (4 * f.FIBER_SEGMENT_MASS * e1.d)
        } else if (f.points[i].la == 0) && (f.points[i].ra == 1) {
            f.points_new[i].vy += e2.s * e2.eps * f.E * f.dt * math.Pow(f.DIAMETER, 2) * math.Pi / (4 * f.FIBER_SEGMENT_MASS * e2.d)
            f.points_new[i].vx += e2.c * e2.eps * f.E * f.dt * math.Pow(f.DIAMETER, 2) * math.Pi / (4 * f.FIBER_SEGMENT_MASS * e2.d)
        } else if (f.points[i].la == 1) && (f.points[i].ra == 1) {
            f.points_new[i].vy += (e1.s * e1.eps / e1.d + e2.s * e2.eps / e2.d) * f.E * f.dt * math.Pow(f.DIAMETER, 2) * math.Pi / (4 * f.FIBER_SEGMENT_MASS)
            f.points_new[i].vx += (e1.c * e1.eps / e1.d + e2.c * e2.eps / e2.d) * f.E * f.dt * math.Pow(f.DIAMETER, 2) * math.Pi / (4 * f.FIBER_SEGMENT_MASS)
        }
    }
}

func (f *Fiber) advance_to_the_next_time_step() {
    for i := 0; i < f.POINTS_NUMBER; i++ {
        f.points[i].vx, f.points[i].vy, f.points[i].ra, f.points[i].la = f.points_new[i].vx, f.points_new[i].vy, f.points_new[i].ra, f.points_new[i].la
    }
    f.time_step_number += 1
    f.T += f.dt
}

func (f *Fiber) step() {
    f.determine_correct_time_step()
    f.handle_fiber_movement()
    f.apply_external_pressure()
    f.handle_elastic_forcese()
    f.advance_to_the_next_time_step()
}

func (f *Fiber) test_fiber() int {
    for i := 0; (i < f.NUMBER_OF_FRAMES) && (f.broken == 0); i++ {
        f.step()
        // fmt.Println(i)
    }
    return f.broken
}

func main() {
    //p := Point{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    //p.move(2.0)


    // 3000,200, 1420,0.01,0, 1,10,1,1,1,115
    // ['NUMBER_OF_FRAMES', 'POINTS_PER_FIBER', 'density', 'diameter', 'is_broken', 'length', 'pressure_amplitude', 'pressure_radius', 'pressure_time', 'strength', 'young']
    // length, diameter, young, density, pressure_time, pressure_radius, pressure_amplitude, strength
    //f := init_fiber(1.0, 0.01, 115, 1420, 1.0, 1.0, 10.0, 1.0)
    //3000,200,1438,0.01,1,10,10,1,1,1,124
    //f := init_fiber(10.0, 0.01, 124, 1438, 1.0, 1.0, 10.0, 1.0)
    // fmt.Println(f.FIBER_SEGMENT_MASS, f.POINTS_NUMBER)
    // fmt.Println(f.points[0])
    // f.points[0].x = 999
    // f.points[1].x = 888
    // fmt.Println(f.points[0])
    // fmt.Println(f.points_new[0], f.points_new[2])

    f := init_fiber(1.0, 0.19, 133.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
    start := time.Now()

    for i := 0; i < 10; i++ {
        f = init_fiber(1.0, 0.19, 133.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
        f.test_fiber()
    }

    elapsed := time.Since(start)
    fmt.Printf("Test fiber took %s\n", elapsed)

}



