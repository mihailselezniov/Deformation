package main

import (
    "fmt"
    "math"
    "time"
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
    t_point                                                                                                       Point
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

    f.points = make([]Point, f.POINTS_NUMBER)
    f.points_new = make([]Point, f.POINTS_NUMBER)
    f.t_point = Point{f.X_0, f.Y_0, 0.0, 0.0, 1.0, 0.0}
    f.points[0], f.points_new[0] = f.t_point, f.t_point
    for i := 1; i < f.POINTS_NUMBER_minus_1; i++ {
        f.t_point = Point{f.X_0 + f.H*float64(i), f.Y_0, 0.0, 0.0, 1.0, 1.0}
        f.points[i], f.points_new[i] = f.t_point, f.t_point
    }
    f.t_point = Point{f.X_0 + f.H*float64(f.POINTS_NUMBER_minus_1), f.Y_0, 0.0, 0.0, 0.0, 1.0}
    f.points[f.POINTS_NUMBER_minus_1], f.points_new[f.POINTS_NUMBER_minus_1] = f.t_point, f.t_point
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




    // start := time.Now()

    // for i := 0; i < 1; i++ {

    //     s1 := time.Now()
    //     f := init_fiber(1.0, 0.01, 115, 1420, 1.0, 1.0, 10.0, 1.0)
    //     fmt.Printf("init_fiber took %s\n", time.Since(s1))

    //     f.test_fiber()
    // }

    // elapsed := time.Since(start)
    // fmt.Printf("Test fiber took %s\n", elapsed)



    // a := 0
    // at := time.Now()
    // for i := 0; i < 100000; i++ {
    //     a ++
    // }
    // fmt.Printf("%s ++ took %s\n", a, time.Since(at))

    // a = 0
    // at2 := time.Now()
    // for i := 0; i < 100000; i++ {
    //     a += 1
    // }
    // fmt.Printf("%s += 1 took %s\n", a, time.Since(at2))

    //fmt.Printf("%s", math.Hypot(3.0, 4.0))


    f := new(Fiber)
    start := time.Now()

    for i := 0; i < 100; i++ {
        f = init_fiber(1.0, 0.19, 133.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
        f.test_fiber()
    }

    elapsed := time.Since(start)
    fmt.Printf("Test fiber took %s\n", elapsed)


f = init_fiber(1.0, 0.19, 133.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("0\n") }
f = init_fiber(10.0, 0.01, 115.0, 1420.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("1\n") }
f = init_fiber(10.0, 0.01, 115.0, 1420.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("2\n") }
f = init_fiber(10.0, 0.01, 115.0, 1429.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("3\n") }
f = init_fiber(10.0, 0.01, 115.0, 1429.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("4\n") }
f = init_fiber(10.0, 0.01, 115.0, 1438.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("5\n") }
f = init_fiber(10.0, 0.01, 115.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("6\n") }
f = init_fiber(10.0, 0.01, 124.0, 1420.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("7\n") }
f = init_fiber(10.0, 0.01, 124.0, 1420.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("8\n") }
f = init_fiber(10.0, 0.01, 124.0, 1429.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("9\n") }
f = init_fiber(10.0, 0.01, 124.0, 1429.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("10\n") }
f = init_fiber(10.0, 0.01, 124.0, 1438.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("11\n") }
f = init_fiber(10.0, 0.01, 124.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("12\n") }
f = init_fiber(10.0, 0.01, 133.0, 1420.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("13\n") }
f = init_fiber(10.0, 0.01, 133.0, 1420.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("14\n") }
f = init_fiber(10.0, 0.01, 133.0, 1429.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("15\n") }
f = init_fiber(10.0, 0.19, 133.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("16\n") }
f = init_fiber(19.0, 0.01, 115.0, 1420.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("17\n") }
f = init_fiber(19.0, 0.01, 115.0, 1420.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("18\n") }
f = init_fiber(19.0, 0.01, 115.0, 1429.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("19\n") }
f = init_fiber(19.0, 0.01, 115.0, 1429.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("20\n") }
f = init_fiber(19.0, 0.01, 115.0, 1438.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("21\n") }
f = init_fiber(19.0, 0.01, 115.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("22\n") }
f = init_fiber(19.0, 0.01, 124.0, 1420.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("23\n") }
f = init_fiber(19.0, 0.01, 124.0, 1420.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("24\n") }
f = init_fiber(19.0, 0.01, 124.0, 1429.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("25\n") }
f = init_fiber(19.0, 0.01, 124.0, 1429.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("26\n") }
f = init_fiber(19.0, 0.01, 124.0, 1438.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("27\n") }
f = init_fiber(19.0, 0.01, 124.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("28\n") }
f = init_fiber(19.0, 0.01, 133.0, 1420.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("29\n") }
f = init_fiber(19.0, 0.01, 133.0, 1420.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("30\n") }
f = init_fiber(19.0, 0.01, 133.0, 1429.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("31\n") }
f = init_fiber(19.0, 0.01, 133.0, 1429.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("32\n") }
f = init_fiber(19.0, 0.01, 133.0, 1438.0, 1.0, 1.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("33\n") }
f = init_fiber(19.0, 0.01, 133.0, 1438.0, 1.0, 1.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("34\n") }
f = init_fiber(19.0, 0.01, 133.0, 1429.0, 1.0, 1.0, 19.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("35\n") }
f = init_fiber(19.0, 0.01, 133.0, 1438.0, 1.0, 1.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("36\n") }
f = init_fiber(19.0, 0.01, 133.0, 1438.0, 1.0, 1.0, 19.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("37\n") }
f = init_fiber(19.0, 0.1, 115.0, 1420.0, 1.0, 1.0, 19.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("38\n") }
f = init_fiber(19.0, 0.1, 115.0, 1420.0, 1.0, 1.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("39\n") }
f = init_fiber(19.0, 0.1, 115.0, 1429.0, 1.0, 1.0, 19.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("40\n") }
f = init_fiber(19.0, 0.1, 115.0, 1429.0, 1.0, 1.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("41\n") }
f = init_fiber(19.0, 0.19, 133.0, 1438.0, 1.0, 1.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("42\n") }
f = init_fiber(1.0, 0.01, 115.0, 1420.0, 1.0, 1.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("43\n") }
f = init_fiber(1.0, 0.01, 115.0, 1420.0, 1.0, 1.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("44\n") }
f = init_fiber(1.0, 0.01, 115.0, 1429.0, 1.0, 1.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("45\n") }
f = init_fiber(1.0, 0.01, 115.0, 1429.0, 1.0, 1.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("46\n") }
f = init_fiber(1.0, 0.01, 115.0, 1438.0, 1.0, 1.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("47\n") }
f = init_fiber(1.0, 0.01, 115.0, 1438.0, 1.0, 1.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("48\n") }
f = init_fiber(1.0, 0.01, 124.0, 1420.0, 1.0, 1.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("49\n") }
f = init_fiber(1.0, 0.01, 124.0, 1420.0, 1.0, 1.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("50\n") }
f = init_fiber(1.0, 0.01, 124.0, 1429.0, 1.0, 1.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("51\n") }
f = init_fiber(1.0, 0.01, 124.0, 1429.0, 1.0, 1.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("52\n") }
f = init_fiber(1.0, 0.01, 124.0, 1438.0, 1.0, 1.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("53\n") }
f = init_fiber(1.0, 0.01, 124.0, 1438.0, 1.0, 1.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("54\n") }
f = init_fiber(1.0, 0.01, 133.0, 1420.0, 1.0, 1.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("55\n") }
f = init_fiber(1.0, 0.01, 133.0, 1420.0, 1.0, 1.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("56\n") }
f = init_fiber(1.0, 0.01, 133.0, 1429.0, 1.0, 1.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("57\n") }
f = init_fiber(1.0, 0.01, 133.0, 1429.0, 1.0, 1.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("58\n") }
f = init_fiber(1.0, 0.01, 133.0, 1438.0, 1.0, 1.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("59\n") }
f = init_fiber(1.0, 0.01, 133.0, 1438.0, 1.0, 1.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("60\n") }
f = init_fiber(19.0, 0.01, 133.0, 1429.0, 19.0, 19.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("61\n") }
f = init_fiber(19.0, 0.01, 133.0, 1429.0, 19.0, 19.0, 19.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("62\n") }
f = init_fiber(19.0, 0.01, 133.0, 1438.0, 19.0, 19.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("63\n") }
f = init_fiber(19.0, 0.01, 133.0, 1438.0, 19.0, 19.0, 19.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("64\n") }
f = init_fiber(19.0, 0.1, 115.0, 1420.0, 19.0, 19.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("65\n") }
f = init_fiber(19.0, 0.1, 115.0, 1420.0, 19.0, 19.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("66\n") }
f = init_fiber(19.0, 0.1, 115.0, 1429.0, 19.0, 19.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("67\n") }
f = init_fiber(19.0, 0.1, 115.0, 1429.0, 19.0, 19.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("68\n") }
f = init_fiber(19.0, 0.1, 115.0, 1438.0, 19.0, 19.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("69\n") }
f = init_fiber(19.0, 0.1, 115.0, 1438.0, 19.0, 19.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("70\n") }
f = init_fiber(19.0, 0.1, 124.0, 1420.0, 19.0, 19.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("71\n") }
f = init_fiber(19.0, 0.1, 124.0, 1420.0, 19.0, 19.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("72\n") }
f = init_fiber(19.0, 0.1, 124.0, 1429.0, 19.0, 19.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("73\n") }
f = init_fiber(19.0, 0.1, 124.0, 1429.0, 19.0, 19.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("74\n") }
f = init_fiber(19.0, 0.1, 124.0, 1438.0, 19.0, 19.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("75\n") }
f = init_fiber(19.0, 0.1, 124.0, 1438.0, 19.0, 19.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("76\n") }
f = init_fiber(19.0, 0.1, 133.0, 1420.0, 19.0, 19.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("77\n") }
f = init_fiber(19.0, 0.1, 133.0, 1420.0, 19.0, 19.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("78\n") }
f = init_fiber(19.0, 0.1, 133.0, 1429.0, 19.0, 19.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("79\n") }
f = init_fiber(19.0, 0.1, 133.0, 1429.0, 19.0, 19.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("80\n") }
f = init_fiber(19.0, 0.1, 133.0, 1438.0, 19.0, 19.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("81\n") }
f = init_fiber(19.0, 0.1, 133.0, 1438.0, 19.0, 19.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("82\n") }
f = init_fiber(10.0, 0.01, 133.0, 1420.0, 19.0, 19.0, 10.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("83\n") }
f = init_fiber(10.0, 0.01, 133.0, 1429.0, 19.0, 19.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("84\n") }
f = init_fiber(10.0, 0.01, 133.0, 1429.0, 19.0, 19.0, 10.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("85\n") }
f = init_fiber(10.0, 0.01, 133.0, 1438.0, 19.0, 19.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("86\n") }
f = init_fiber(10.0, 0.01, 133.0, 1438.0, 19.0, 19.0, 10.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("87\n") }
f = init_fiber(10.0, 0.1, 115.0, 1420.0, 19.0, 19.0, 10.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("88\n") }
f = init_fiber(10.0, 0.1, 115.0, 1420.0, 19.0, 19.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("89\n") }
f = init_fiber(10.0, 0.1, 115.0, 1429.0, 19.0, 19.0, 10.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("90\n") }
f = init_fiber(10.0, 0.1, 115.0, 1429.0, 19.0, 19.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("91\n") }
f = init_fiber(10.0, 0.1, 115.0, 1438.0, 19.0, 19.0, 10.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("92\n") }
f = init_fiber(10.0, 0.1, 133.0, 1438.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("93\n") }
f = init_fiber(10.0, 0.19, 115.0, 1420.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("94\n") }
f = init_fiber(10.0, 0.19, 115.0, 1420.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("95\n") }
f = init_fiber(10.0, 0.19, 115.0, 1429.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("96\n") }
f = init_fiber(10.0, 0.19, 115.0, 1429.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("97\n") }
f = init_fiber(10.0, 0.19, 115.0, 1438.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("98\n") }
f = init_fiber(10.0, 0.19, 115.0, 1438.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("99\n") }
f = init_fiber(10.0, 0.19, 124.0, 1420.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("100\n") }
f = init_fiber(10.0, 0.19, 124.0, 1420.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("101\n") }
f = init_fiber(10.0, 0.19, 124.0, 1429.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("102\n") }
f = init_fiber(10.0, 0.19, 124.0, 1429.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("103\n") }
f = init_fiber(10.0, 0.19, 124.0, 1438.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("104\n") }
f = init_fiber(10.0, 0.19, 124.0, 1438.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("105\n") }
f = init_fiber(10.0, 0.19, 133.0, 1420.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("106\n") }
f = init_fiber(10.0, 0.19, 133.0, 1420.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("107\n") }
f = init_fiber(10.0, 0.19, 133.0, 1429.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("108\n") }
f = init_fiber(10.0, 0.19, 133.0, 1429.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("109\n") }
f = init_fiber(10.0, 0.19, 133.0, 1438.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("110\n") }
f = init_fiber(10.0, 0.19, 133.0, 1438.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("111\n") }
f = init_fiber(19.0, 0.01, 115.0, 1420.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("112\n") }
f = init_fiber(19.0, 0.01, 115.0, 1420.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("113\n") }
f = init_fiber(1.0, 0.19, 133.0, 1420.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("114\n") }
f = init_fiber(1.0, 0.19, 133.0, 1429.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("115\n") }
f = init_fiber(1.0, 0.19, 133.0, 1429.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("116\n") }
f = init_fiber(1.0, 0.19, 133.0, 1438.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("117\n") }
f = init_fiber(1.0, 0.19, 133.0, 1438.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("118\n") }
f = init_fiber(10.0, 0.01, 115.0, 1420.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("119\n") }
f = init_fiber(10.0, 0.01, 115.0, 1420.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("120\n") }
f = init_fiber(10.0, 0.01, 115.0, 1429.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("121\n") }
f = init_fiber(10.0, 0.01, 115.0, 1429.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("122\n") }
f = init_fiber(10.0, 0.01, 115.0, 1438.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("123\n") }
f = init_fiber(10.0, 0.01, 115.0, 1438.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("124\n") }
f = init_fiber(19.0, 0.19, 115.0, 1420.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("125\n") }
f = init_fiber(19.0, 0.19, 115.0, 1420.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("126\n") }
f = init_fiber(19.0, 0.19, 115.0, 1429.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("127\n") }
f = init_fiber(19.0, 0.19, 115.0, 1429.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("128\n") }
f = init_fiber(19.0, 0.19, 115.0, 1438.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("129\n") }
f = init_fiber(19.0, 0.19, 115.0, 1438.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("130\n") }
f = init_fiber(19.0, 0.19, 124.0, 1420.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("131\n") }
f = init_fiber(19.0, 0.19, 124.0, 1420.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("132\n") }
f = init_fiber(19.0, 0.19, 124.0, 1429.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("133\n") }
f = init_fiber(19.0, 0.19, 124.0, 1429.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("134\n") }
f = init_fiber(19.0, 0.19, 124.0, 1438.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("135\n") }
f = init_fiber(19.0, 0.19, 124.0, 1438.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("136\n") }
f = init_fiber(19.0, 0.19, 133.0, 1420.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("137\n") }
f = init_fiber(19.0, 0.19, 133.0, 1420.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("138\n") }
f = init_fiber(19.0, 0.19, 133.0, 1429.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("139\n") }
f = init_fiber(19.0, 0.19, 133.0, 1429.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("140\n") }
f = init_fiber(19.0, 0.19, 133.0, 1438.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("141\n") }
f = init_fiber(19.0, 0.19, 133.0, 1438.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("142\n") }
f = init_fiber(1.0, 0.01, 115.0, 1420.0, 19.0, 10.0, 28.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("143\n") }
f = init_fiber(1.0, 0.01, 115.0, 1420.0, 19.0, 10.0, 28.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("144\n") }
f = init_fiber(10.0, 0.19, 133.0, 1429.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("145\n") }
f = init_fiber(10.0, 0.19, 133.0, 1429.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("146\n") }
f = init_fiber(10.0, 0.19, 133.0, 1438.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("147\n") }
f = init_fiber(10.0, 0.19, 133.0, 1438.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("148\n") }
f = init_fiber(19.0, 0.01, 115.0, 1420.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("149\n") }
f = init_fiber(19.0, 0.01, 115.0, 1420.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("150\n") }
f = init_fiber(19.0, 0.01, 115.0, 1429.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("151\n") }
f = init_fiber(19.0, 0.01, 115.0, 1429.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("152\n") }
f = init_fiber(10.0, 0.01, 133.0, 1438.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("153\n") }
f = init_fiber(10.0, 0.1, 115.0, 1420.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("154\n") }
f = init_fiber(10.0, 0.1, 115.0, 1420.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("155\n") }
f = init_fiber(10.0, 0.1, 115.0, 1429.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("156\n") }
f = init_fiber(10.0, 0.1, 115.0, 1429.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("157\n") }
f = init_fiber(10.0, 0.1, 115.0, 1438.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("158\n") }
f = init_fiber(10.0, 0.1, 115.0, 1438.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("159\n") }
f = init_fiber(10.0, 0.1, 124.0, 1420.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("160\n") }
f = init_fiber(10.0, 0.1, 124.0, 1420.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("161\n") }
f = init_fiber(10.0, 0.1, 124.0, 1429.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("162\n") }
f = init_fiber(10.0, 0.1, 124.0, 1429.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("163\n") }
f = init_fiber(10.0, 0.1, 124.0, 1438.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("164\n") }
f = init_fiber(10.0, 0.1, 124.0, 1438.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("165\n") }
f = init_fiber(10.0, 0.1, 133.0, 1420.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("166\n") }
f = init_fiber(10.0, 0.1, 133.0, 1420.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("167\n") }
f = init_fiber(10.0, 0.1, 133.0, 1429.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("168\n") }
f = init_fiber(10.0, 0.1, 133.0, 1429.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("169\n") }
f = init_fiber(10.0, 0.1, 133.0, 1438.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("170\n") }
f = init_fiber(10.0, 0.1, 133.0, 1438.0, 19.0, 10.0, 19.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("171\n") }
f = init_fiber(10.0, 0.19, 115.0, 1420.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("172\n") }
f = init_fiber(19.0, 0.1, 115.0, 1420.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("173\n") }
f = init_fiber(19.0, 0.1, 115.0, 1420.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("174\n") }
f = init_fiber(19.0, 0.1, 115.0, 1429.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("175\n") }
f = init_fiber(19.0, 0.1, 115.0, 1429.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("176\n") }
f = init_fiber(19.0, 0.1, 115.0, 1438.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("177\n") }
f = init_fiber(19.0, 0.1, 115.0, 1438.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("178\n") }
f = init_fiber(19.0, 0.1, 124.0, 1420.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("179\n") }
f = init_fiber(19.0, 0.1, 124.0, 1420.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("180\n") }
f = init_fiber(19.0, 0.1, 124.0, 1429.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("181\n") }
f = init_fiber(19.0, 0.1, 124.0, 1429.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("182\n") }
f = init_fiber(19.0, 0.1, 124.0, 1438.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("183\n") }
f = init_fiber(19.0, 0.1, 124.0, 1438.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("184\n") }
f = init_fiber(19.0, 0.1, 133.0, 1420.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("185\n") }
f = init_fiber(19.0, 0.1, 133.0, 1420.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("186\n") }
f = init_fiber(19.0, 0.1, 133.0, 1429.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("187\n") }
f = init_fiber(19.0, 0.1, 133.0, 1429.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 1.0 { fmt.Printf("188\n") }
f = init_fiber(19.0, 0.1, 133.0, 1438.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("189\n") }
f = init_fiber(19.0, 0.1, 133.0, 1438.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("190\n") }
f = init_fiber(19.0, 0.19, 115.0, 1420.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("191\n") }
f = init_fiber(19.0, 0.19, 115.0, 1420.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("192\n") }
f = init_fiber(19.0, 0.19, 115.0, 1429.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("193\n") }
f = init_fiber(19.0, 0.19, 115.0, 1429.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("194\n") }
f = init_fiber(19.0, 0.19, 115.0, 1438.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("195\n") }
f = init_fiber(19.0, 0.19, 115.0, 1438.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("196\n") }
f = init_fiber(19.0, 0.19, 124.0, 1420.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("197\n") }
f = init_fiber(19.0, 0.19, 124.0, 1420.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("198\n") }
f = init_fiber(19.0, 0.19, 124.0, 1429.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("199\n") }
f = init_fiber(19.0, 0.19, 124.0, 1429.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("200\n") }
f = init_fiber(19.0, 0.19, 124.0, 1438.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("201\n") }
f = init_fiber(19.0, 0.19, 124.0, 1438.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("202\n") }
f = init_fiber(19.0, 0.19, 133.0, 1420.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("203\n") }
f = init_fiber(19.0, 0.19, 133.0, 1420.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("204\n") }
f = init_fiber(19.0, 0.19, 133.0, 1429.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("205\n") }
f = init_fiber(19.0, 0.19, 133.0, 1429.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("206\n") }
f = init_fiber(19.0, 0.19, 133.0, 1438.0, 19.0, 10.0, 10.0, 1.0)
if f.test_fiber() != 1.0 { fmt.Printf("207\n") }
f = init_fiber(19.0, 0.19, 133.0, 1438.0, 19.0, 10.0, 10.0, 10.0)
if f.test_fiber() != 0.0 { fmt.Printf("208\n") }
f = init_fiber(1.0, 0.01, 115.0, 1420.0, 19.0, 10.0, 19.0, 1.0)
if f.test_fiber() != 0.0 { fmt.Printf("209\n") }



}



