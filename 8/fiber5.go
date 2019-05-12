package main

import (
    "fmt"
    "math"
    "time"
    "strings"
    "strconv"
)

type Elastic struct {
    d, eps, s, c float64
}

type Point struct {
    x, y, vx, vy, ra, la float64
}

func (p *Point) move(dt float64) {
    p.x += p.vx * dt
    p.y += p.vy * dt
}

type Fiber struct {
    LENGTH, DIAMETER, E, DENSITY, H, PRESSURE_T, PRESSURE_R, PRESSURE_AMPLITUDE, DEFORMATION_LIMIT float64
    time_step_number, FIBER_SEGMENT_MASS, T, X_0, Y_0, dt, P_wave_speed, DIAMETER2_Pi float64
    TEMP_apply_forces_numerator, TEMP_apply_forces_denominator, TEMP_apply_forces, TEMP_apply_forces_fracture float64
    TEMP_apply_pressure_AmpDimHMass, TEMP_apply_pressure_Pi2R float64
    TEMP_neighbour_x, TEMP_neighbour_y, TEMP_eps_fiber_fracture float64
    POINTS_NUMBER, POINTS_NUMBER_minus_1, POINTS_PER_FIBER, NUMBER_OF_FRAMES, TEMP_apply_pressure_points2, broken int
    points, points_new []Point
    e1, e2 Elastic
}

func init_fiber(length, diameter, young, density, pressure_time, pressure_radius, pressure_amplitude, strength float64) *Fiber {
    f := new(Fiber)

    // numerical solver params
    f.POINTS_PER_FIBER, f.NUMBER_OF_FRAMES = 200, 3000

    f.LENGTH, f.DIAMETER, f.E, f.DENSITY, f.H, f.PRESSURE_T, f.PRESSURE_R, f.PRESSURE_AMPLITUDE, f.DEFORMATION_LIMIT =
      length / 100.0, // fiber length, in centimeters
      diameter / 1000.0, // fiber diameter, in millimeters
      young * 1000000000, // fiber young modulus, in GPa
      density, // fiber density, in kg/m3
      (length / 100.0) / float64(f.POINTS_PER_FIBER), // space step
      pressure_time / 1000000, // pressure pulse total time, in microseconds
      pressure_radius / 100.0, // pressure pulse radius, in centimeters
      pressure_amplitude * 1000000, // pressure pulse amplitude, in MPa
      strength / 100.0 // fiber deformation limit, in percents

    // calculate ans store these internal params from passed values
    f.DIAMETER2_Pi = math.Pow(f.DIAMETER, 2) * math.Pi
    f.FIBER_SEGMENT_MASS = f.DENSITY * (0.25 * f.DIAMETER2_Pi * f.H)
    f.POINTS_NUMBER_minus_1 = int(f.LENGTH / f.H)
    f.POINTS_NUMBER = f.POINTS_NUMBER_minus_1 + 1
    f.TEMP_apply_pressure_points2, f.TEMP_apply_pressure_AmpDimHMass = f.POINTS_NUMBER / 2, (f.PRESSURE_AMPLITUDE * f.DIAMETER * f.H) / f.FIBER_SEGMENT_MASS
    f.TEMP_apply_forces_numerator, f.TEMP_apply_forces_denominator = f.E * f.DIAMETER2_Pi, 4 * f.FIBER_SEGMENT_MASS
    f.TEMP_apply_forces = f.TEMP_apply_forces_numerator / f.TEMP_apply_forces_denominator
    f.P_wave_speed = math.Sqrt(f.E / f.DENSITY)
    f.TEMP_apply_pressure_Pi2R = math.Pi / (f.PRESSURE_R * 2.0)
    f.TEMP_eps_fiber_fracture = f.DEFORMATION_LIMIT * f.H

    f.points = append(f.points, Point{f.X_0, f.Y_0, 0.0, 0.0, 1.0, 0.0})
    for i := 1; i < f.POINTS_NUMBER_minus_1; i++ {
        f.points = append(f.points, Point{f.X_0 + f.H * float64(i), f.Y_0, 0.0, 0.0, 1.0, 1.0})
    }
    f.points = append(f.points, Point{f.X_0 + f.H * float64(f.POINTS_NUMBER_minus_1), f.Y_0, 0.0, 0.0, 0.0, 1.0})
    f.points_new = make([]Point, len(f.points))
    copy(f.points_new, f.points)
    return f
}

func (f *Fiber) get_tau(i int) float64 {
    return math.Hypot(f.points[i-1].x - f.points[i].x, f.points[i-1].y - f.points[i].y) / f.P_wave_speed // !!!
}

func (f *Fiber) determine_correct_time_step() {
    f.dt = f.get_tau(1)
    for i := 2; i < f.POINTS_NUMBER; i++ {
        //if f.points[i].la == 0 {continue}
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
    dist := math.Abs(f.points[n].x - f.points[f.TEMP_apply_pressure_points2].x)
    if dist <= f.PRESSURE_R {
        return f.TEMP_apply_pressure_AmpDimHMass * f.dt * math.Pow(math.Cos(dist * f.TEMP_apply_pressure_Pi2R), 2)
    }
    return 0.0
}

func (f *Fiber) apply_external_pressure() {
    for i := 0; i < f.POINTS_NUMBER; i++ {
        f.points_new[i].vy += f.apply_pressure(i)
    }
}

func (f *Fiber) calculate_force_from_neighbour(i_neighbour, i int, e *Elastic) {
    f.TEMP_neighbour_x, f.TEMP_neighbour_y = f.points[i_neighbour].x - f.points[i].x, f.points[i_neighbour].y - f.points[i].y // !!!
    e.d = math.Hypot(f.TEMP_neighbour_x, f.TEMP_neighbour_y) // !!!
    e.eps = e.d - f.H
    e.s = f.TEMP_neighbour_y / e.d
    e.c = f.TEMP_neighbour_x / e.d
}

func (f *Fiber) handle_elastic_forcese() {
    for i := 0; i < f.POINTS_NUMBER; i++ {
        f.e1.d, f.e1.eps, f.e1.s, f.e1.c, f.e2.d, f.e2.eps, f.e2.s, f.e2.c = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        // calculate force from left and right neighbour
        if i >= 1 {f.calculate_force_from_neighbour(i-1, i, &f.e1)}
        if i < f.POINTS_NUMBER_minus_1 {f.calculate_force_from_neighbour(i+1, i, &f.e2)}

        // handle possible fiber fracture
        if f.e1.eps >= f.TEMP_eps_fiber_fracture {f.points_new[i].la, f.broken = 0.0, 1}
        if f.e2.eps >= f.TEMP_eps_fiber_fracture {f.points_new[i].ra, f.broken = 0.0, 1}

        // apply forces taking possible fracture into account
        if (f.points[i].ra == 0) && (f.points[i].la == 1) {
            f.TEMP_apply_forces_fracture = (f.e1.eps * f.dt * f.TEMP_apply_forces_numerator) / (f.TEMP_apply_forces_denominator * f.e1.d)
            f.points_new[i].vy += f.e1.s * f.TEMP_apply_forces_fracture
            f.points_new[i].vx += f.e1.c * f.TEMP_apply_forces_fracture
        } else if (f.points[i].la == 0) && (f.points[i].ra == 1) {
            f.TEMP_apply_forces_fracture = (f.e2.eps * f.dt * f.TEMP_apply_forces_numerator) / (f.TEMP_apply_forces_denominator * f.e2.d)
            f.points_new[i].vy += f.e2.s * f.TEMP_apply_forces_fracture
            f.points_new[i].vx += f.e2.c * f.TEMP_apply_forces_fracture
        } else if (f.points[i].la == 1) && (f.points[i].ra == 1) {
            t1, t2, t3 := f.e1.eps / f.e1.d, f.e2.eps / f.e2.d, f.dt * f.TEMP_apply_forces
            f.points_new[i].vy += (f.e1.s * t1 + f.e2.s * t2) * t3
            f.points_new[i].vx += (f.e1.c * t1 + f.e2.c * t2) * t3
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
    }
    return f.broken
}

func parse_par(s string) []float64 {
    par := strings.Split(s, ",")
    p := make([]float64, len(par))
    for index, element := range par {
        elem, err := strconv.ParseFloat(element, 64)
        if err != nil {fmt.Printf("%s\n", err)}
        p[index] = elem
    }
    return p
}

func main() {
    f := new(Fiber)

    f = init_fiber(1.0, 0.19, 133.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
    fmt.Printf("%s\n", f.broken)

    start := time.Now()

    s := "100.0,0.5,300.0,2000.0,100.0"
    p := parse_par(s)

    pressure_radius := [10]float64{0.0, 0.56, 1.11, 1.67, 2.22, 2.78, 3.33, 3.89, 4.44, 5.0}
    pressure_amplitude := [10]float64{0.0, 22.22, 44.44, 66.67, 88.89, 111.11, 133.33, 155.56, 177.78, 200.0}
    strength := [10]float64{0.2, 1.29, 2.38, 3.47, 4.56, 5.64, 6.73, 7.82, 8.91, 10.0}
    var broken_arr [1000]int

    for pri, pr := range pressure_radius {
        for pai, pa := range pressure_amplitude {
            for si, s := range strength {
                f = init_fiber(p[0], p[1], p[2], p[3], p[4], pr, pa, s)
                broken_arr[pri*100+pai*10+si] = f.test_fiber()
                //break
            }
            //break
        }
        //break
    }

    fmt.Printf("%s\n", p)
    fmt.Printf("%s\n", strings.Trim(strings.Join(strings.Fields(fmt.Sprint(broken_arr)), ""), "[]"))

    current_state, count_state := 0, 0
    var count_state_arr []int
    for _, state := range broken_arr {
        if current_state == state {
            count_state += 1
        } else {
            count_state_arr = append(count_state_arr, count_state)
            current_state, count_state = state, 1
        }
    }
    count_state_arr = append(count_state_arr, count_state)

    res := strings.Trim(strings.Join(strings.Fields(fmt.Sprint(count_state_arr)), ","), "[]")

    fmt.Printf("%s\n", res)

    // for i := 0; i < 1000; i++ {
    //     f = init_fiber(1.0, 0.19, 133.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
    //     f.test_fiber()
    // }

    elapsed := time.Since(start)
    fmt.Printf("Test fiber took %s\n", elapsed)



}



