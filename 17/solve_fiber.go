package main

import (
    "fmt"
    "os"
    "math"
    "strconv"
    "strings"
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
    LENGTH, DIAMETER, E, DENSITY, H, PRESSURE_T, PRESSURE_R, PRESSURE_AMPLITUDE, DEFORMATION_LIMIT                float64
    time_step_number, FIBER_SEGMENT_MASS, T, X_0, Y_0, dt, P_wave_speed, DIAMETER2_Pi                             float64
    TEMP_apply_forces_numerator, TEMP_apply_forces_denominator, TEMP_apply_forces, TEMP_apply_forces_fracture     float64
    TEMP_apply_pressure_AmpDimHMass, TEMP_apply_pressure_Pi2R                                                     float64
    TEMP_neighbour_x, TEMP_neighbour_y, TEMP_eps_fiber_fracture                                                   float64
    POINTS_NUMBER, POINTS_NUMBER_minus_1, POINTS_PER_FIBER, NUMBER_OF_FRAMES, TEMP_apply_pressure_points2, broken int
    points, points_new                                                                                            []Point
    e1, e2                                                                                                        Elastic
}

func init_fiber(length, diameter, young, density, pressure_time, pressure_radius, pressure_amplitude, strength float64) *Fiber {
    f := new(Fiber)

    // numerical solver params
    f.POINTS_PER_FIBER, f.NUMBER_OF_FRAMES = 200, 3000

    f.LENGTH, f.DIAMETER, f.E, f.DENSITY, f.H, f.PRESSURE_T, f.PRESSURE_R, f.PRESSURE_AMPLITUDE, f.DEFORMATION_LIMIT =
        length/100.0, // fiber length, in centimeters
        diameter/1000.0, // fiber diameter, in millimeters
        young*1000000000, // fiber young modulus, in GPa
        density, // fiber density, in kg/m3
        (length/100.0)/float64(f.POINTS_PER_FIBER), // space step
        pressure_time/1000000, // pressure pulse total time, in microseconds
        pressure_radius/100.0, // pressure pulse radius, in centimeters
        pressure_amplitude*1000000, // pressure pulse amplitude, in MPa
        strength/100.0 // fiber deformation limit, in percents

    // calculate ans store these internal params from passed values
    f.DIAMETER2_Pi = math.Pow(f.DIAMETER, 2) * math.Pi
    f.FIBER_SEGMENT_MASS = f.DENSITY * (0.25 * f.DIAMETER2_Pi * f.H)
    f.POINTS_NUMBER_minus_1 = int(f.LENGTH / f.H)
    f.POINTS_NUMBER = f.POINTS_NUMBER_minus_1 + 1
    f.TEMP_apply_pressure_points2, f.TEMP_apply_pressure_AmpDimHMass = f.POINTS_NUMBER/2, (f.PRESSURE_AMPLITUDE*f.DIAMETER*f.H)/f.FIBER_SEGMENT_MASS
    f.TEMP_apply_forces_numerator, f.TEMP_apply_forces_denominator = f.E*f.DIAMETER2_Pi, 4*f.FIBER_SEGMENT_MASS
    f.TEMP_apply_forces = f.TEMP_apply_forces_numerator / f.TEMP_apply_forces_denominator
    f.P_wave_speed = math.Sqrt(f.E / f.DENSITY)
    f.TEMP_apply_pressure_Pi2R = math.Pi / (f.PRESSURE_R * 2.0)
    f.TEMP_eps_fiber_fracture = f.DEFORMATION_LIMIT * f.H

    f.points = append(f.points, Point{f.X_0, f.Y_0, 0.0, 0.0, 1.0, 0.0})
    for i := 1; i < f.POINTS_NUMBER_minus_1; i++ {
        f.points = append(f.points, Point{f.X_0 + f.H*float64(i), f.Y_0, 0.0, 0.0, 1.0, 1.0})
    }
    f.points = append(f.points, Point{f.X_0 + f.H*float64(f.POINTS_NUMBER_minus_1), f.Y_0, 0.0, 0.0, 0.0, 1.0})
    f.points_new = make([]Point, len(f.points))
    copy(f.points_new, f.points)
    return f
}

func (f *Fiber) get_tau(i int) float64 {
    return math.Hypot(f.points[i-1].x-f.points[i].x, f.points[i-1].y-f.points[i].y) / f.P_wave_speed
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
    if f.T >= f.PRESSURE_T {
        return 0.0
    }
    dist := math.Abs(f.points[n].x - f.points[f.TEMP_apply_pressure_points2].x)
    if dist <= f.PRESSURE_R {
        return f.TEMP_apply_pressure_AmpDimHMass * f.dt * math.Pow(math.Cos(dist*f.TEMP_apply_pressure_Pi2R), 2)
    }
    return 0.0
}

func (f *Fiber) apply_external_pressure() {
    for i := 0; i < f.POINTS_NUMBER; i++ {
        f.points_new[i].vy += f.apply_pressure(i)
    }
}

func (f *Fiber) calculate_force_from_neighbour(i_neighbour, i int, e *Elastic) {
    f.TEMP_neighbour_x, f.TEMP_neighbour_y = f.points[i_neighbour].x-f.points[i].x, f.points[i_neighbour].y-f.points[i].y
    e.d = math.Hypot(f.TEMP_neighbour_x, f.TEMP_neighbour_y)
    e.eps = e.d - f.H
    e.s = f.TEMP_neighbour_y / e.d
    e.c = f.TEMP_neighbour_x / e.d
}

func (f *Fiber) handle_elastic_forcese() {
    for i := 0; i < f.POINTS_NUMBER; i++ {
        f.e1.d, f.e1.eps, f.e1.s, f.e1.c, f.e2.d, f.e2.eps, f.e2.s, f.e2.c = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        // calculate force from left and right neighbour
        if i >= 1 {
            f.calculate_force_from_neighbour(i-1, i, &f.e1)
        }
        if i < f.POINTS_NUMBER_minus_1 {
            f.calculate_force_from_neighbour(i+1, i, &f.e2)
        }

        // handle possible fiber fracture
        if f.e1.eps >= f.TEMP_eps_fiber_fracture {
            f.points_new[i].la, f.broken = 0.0, 1
        }
        if f.e2.eps >= f.TEMP_eps_fiber_fracture {
            f.points_new[i].ra, f.broken = 0.0, 1
        }

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
            t1, t2, t3 := f.e1.eps/f.e1.d, f.e2.eps/f.e2.d, f.dt*f.TEMP_apply_forces
            f.points_new[i].vy += (f.e1.s*t1 + f.e2.s*t2) * t3
            f.points_new[i].vx += (f.e1.c*t1 + f.e2.c*t2) * t3
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

func save_to_file(s string) {
    arg := os.Args[1]
    f_name := arg + ".txt"
    f, err := os.Create(f_name)
    if err != nil {
        fmt.Println(err)
        return
    }
    l, err := f.WriteString(s)
    if err != nil {
        fmt.Println(err)
        f.Close()
        return
    }
    fmt.Println(l, "bytes written successfully to " + f_name)
    err = f.Close()
    if err != nil {
        fmt.Println(err)
        return
    }
}

func start() {
    f := new(Fiber)


    arg := os.Args[1]
    s_arg := strings.Split(arg, ",")
    fmt.Println(s_arg)
    // 15.0,0.04,206.67,1611.11,27.78,4.72,11.11,9.46

    var f_arg []float64
    for _, arg := range s_arg {
        if n, err := strconv.ParseFloat(arg, 64); err == nil {
            f_arg = append(f_arg, n)
        }
    }
    fmt.Println(f_arg)
    f = init_fiber(f_arg[0], f_arg[1], f_arg[2], f_arg[3], f_arg[4], f_arg[5], f_arg[6], f_arg[7])
    fmt.Println("#"+arg+" ", f.test_fiber())
    return






    i_length, err := strconv.Atoi(arg)
    if err != nil {
        // handle error
        fmt.Println(err)
        os.Exit(2)
    }
    length := [9]float64{ 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0 }
    diameter := [9]float64{ 0.04, 0.09, 0.15, 0.2, 0.25, 0.31, 0.36, 0.42, 0.47 }
    young := [9]float64{ 73.33, 100.0, 126.67, 153.33, 180.0, 206.67, 233.33, 260.0, 286.67 }
    density := [9]float64{ 1055.56, 1166.67, 1277.78, 1388.89, 1500.0, 1611.11, 1722.22, 1833.33, 1944.44 }
    pressure_time := [9]float64{ 5.56, 16.67, 27.78, 38.89, 50.0, 61.11, 72.22, 83.33, 94.44 }
    pressure_radius := [9]float64{ 0.28, 0.83, 1.39, 1.94, 2.5, 3.06, 3.61, 4.17, 4.72 }
    pressure_amplitude := [9]float64{ 11.11, 33.33, 55.56, 77.78, 100.0, 122.22, 144.44, 166.67, 188.89 }
    strength := [9]float64{ 0.74, 1.83, 2.92, 4.01, 5.1, 6.19, 7.28, 8.37, 9.46 }
    var broken_arr [4782969]int

    broken_i := 0
    for dii, di := range diameter {
        for _, yo := range young {
            for _, de := range density {
                for _, pt := range pressure_time {
                    for _, pr := range pressure_radius {
                        for _, pa := range pressure_amplitude {
                            for _, s := range strength {
                                f = init_fiber(length[i_length], di, yo, de, pt, pr, pa, s)
                                broken_arr[broken_i] = f.test_fiber()
                                broken_i++
                            }
                        }
                    }
                }
            }
        }
        fmt.Println("#"+arg+" ", dii)
    }

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
    save_to_file(res)
}

func main() {
    start()
}
