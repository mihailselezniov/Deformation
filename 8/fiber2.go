package main

import (
    "fmt"
    "math"
    "time"
    //"reflect"
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
    time_step_number, FIBER_SEGMENT_MASS, T, X_0, Y_0, dt, P_wave_speed float64
    TEMP_apply_forces_numerator, TEMP_apply_forces_denominator float64
    TEMP_apply_pressure_AmpDimHMass, TEMP_apply_pressure_Pi2R float64
    TEMP_neighbour_x, TEMP_neighbour_y float64
    POINTS_NUMBER, POINTS_PER_FIBER, NUMBER_OF_FRAMES, TEMP_apply_pressure_points2, broken int
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
    f.TEMP_apply_pressure_points2, f.TEMP_apply_pressure_AmpDimHMass = f.POINTS_NUMBER / 2, (f.PRESSURE_AMPLITUDE * f.DIAMETER * f.H) / f.FIBER_SEGMENT_MASS
    f.TEMP_apply_forces_numerator, f.TEMP_apply_forces_denominator = f.E * math.Pow(f.DIAMETER, 2) * math.Pi, 4 * f.FIBER_SEGMENT_MASS
    f.P_wave_speed = math.Sqrt(f.E / f.DENSITY)
    f.TEMP_apply_pressure_Pi2R = math.Pi / (f.PRESSURE_R * 2.0)

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
    return math.Hypot(f.points[i-1].x - f.points[i].x, f.points[i-1].y - f.points[i].y) / f.P_wave_speed // !!!
}

func (f *Fiber) determine_correct_time_step() {
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
    t := time.Now()
    t0, t1, t2, t3 := time.Since(t), time.Since(t), time.Since(t), time.Since(t)
    for i := 0; i < f.POINTS_NUMBER; i++ {
        s50 := time.Now()
        //var e1, e2 Elastic
        f.e1.d, f.e1.eps, f.e1.s, f.e1.c, f.e2.d, f.e2.eps, f.e2.s, f.e2.c = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        t0 += time.Since(s50)

        s51 := time.Now()
        // calculate force from left and right neighbour
        if i >= 1 {f.calculate_force_from_neighbour(i-1, i, &f.e1)}
        if i < f.POINTS_NUMBER - 1 {f.calculate_force_from_neighbour(i+1, i, &f.e2)}
        t1 += time.Since(s51)

        s52 := time.Now()
        // handle possible fiber fracture
        t_eps := f.DEFORMATION_LIMIT * f.H // !!!
        if f.e1.eps >= t_eps {f.points_new[i].la, f.broken = 0.0, 1}
        if f.e2.eps >= t_eps {f.points_new[i].ra, f.broken = 0.0, 1}
        t2 += time.Since(s52)

        s53 := time.Now()
        // apply forces taking possible fracture into account
        if (f.points[i].ra == 0) && (f.points[i].la == 1) {
            temp := (f.e1.eps * f.dt * f.TEMP_apply_forces_numerator) / (f.TEMP_apply_forces_denominator * f.e1.d)
            f.points_new[i].vy += f.e1.s * temp
            f.points_new[i].vx += f.e1.c * temp
        } else if (f.points[i].la == 0) && (f.points[i].ra == 1) {
            temp := (f.e2.eps * f.dt * f.TEMP_apply_forces_numerator) / (f.TEMP_apply_forces_denominator * f.e2.d)
            f.points_new[i].vy += f.e2.s * temp
            f.points_new[i].vx += f.e2.c * temp
        } else if (f.points[i].la == 1) && (f.points[i].ra == 1) {
            t1, t2, t3 := f.e1.eps / f.e1.d, f.e2.eps / f.e2.d, f.dt * f.TEMP_apply_forces_numerator / f.TEMP_apply_forces_denominator
            f.points_new[i].vy += (f.e1.s * t1 + f.e2.s * t2) * t3
            f.points_new[i].vx += (f.e1.c * t1 + f.e2.c * t2) * t3
        }
        t3 += time.Since(s53)
    }
    fmt.Printf("  s50 took %s\n", t0)
    fmt.Printf("  s51 took %s\n", t1)
    fmt.Printf("  s52 took %s\n", t2)
    fmt.Printf("  s53 took %s\n", t3)
}

func (f *Fiber) advance_to_the_next_time_step() {
    for i := 0; i < f.POINTS_NUMBER; i++ {
        f.points[i].vx, f.points[i].vy, f.points[i].ra, f.points[i].la = f.points_new[i].vx, f.points_new[i].vy, f.points_new[i].ra, f.points_new[i].la
    }
    f.time_step_number += 1
    f.T += f.dt
}

func (f *Fiber) step() {
    s2 := time.Now()
    f.determine_correct_time_step()
    fmt.Printf("determine_correct_time_step took %s\n", time.Since(s2))
    s3 := time.Now()
    f.handle_fiber_movement()
    fmt.Printf("handle_fiber_movement took %s\n", time.Since(s3))
    s4 := time.Now()
    f.apply_external_pressure()
    fmt.Printf("apply_external_pressure took %s\n", time.Since(s4))
    s5 := time.Now()
    f.handle_elastic_forcese()
    fmt.Printf("handle_elastic_forcese took %s\n", time.Since(s5))
    s6 := time.Now()
    f.advance_to_the_next_time_step()
    fmt.Printf("advance_to_the_next_time_step took %s\n\n", time.Since(s6))
}

func (f *Fiber) test_fiber() int {
    for i := 0; (i < f.NUMBER_OF_FRAMES) && (f.broken == 0); i++ {
        f.step()
        break
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


    start := time.Now()

    for i := 0; i < 1; i++ {

        s1 := time.Now()
        f := init_fiber(1.0, 0.01, 115, 1420, 1.0, 1.0, 10.0, 1.0)
        fmt.Printf("init_fiber took %s\n", time.Since(s1))

        f.test_fiber()
    }

    elapsed := time.Since(start)
    fmt.Printf("Test fiber took %s\n", elapsed)

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

}



