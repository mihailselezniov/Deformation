package main

import (
    "fmt"
    "os"
    "strconv"
    "strings"
)

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

func main() {

    arg := os.Args[1]
    i_length, err := strconv.Atoi(arg)
    if err != nil {
        // handle error
        fmt.Println(err)
        os.Exit(2)
    }
    length := [9]float64{ 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0 }
    //diameter := [9]float64{ 0.04, 0.09, 0.15, 0.2, 0.25, 0.31, 0.36, 0.42, 0.47 }
    //young := [9]float64{ 73.33, 100.0, 126.67, 153.33, 180.0, 206.67, 233.33, 260.0, 286.67 }
    //density := [9]float64{ 1055.56, 1166.67, 1277.78, 1388.89, 1500.0, 1611.11, 1722.22, 1833.33, 1944.44 }
    //pressure_time := [9]float64{ 5.56, 16.67, 27.78, 38.89, 50.0, 61.11, 72.22, 83.33, 94.44 }
    //pressure_radius := [9]float64{ 0.28, 0.83, 1.39, 1.94, 2.5, 3.06, 3.61, 4.17, 4.72 }
    //pressure_amplitude := [9]float64{ 11.11, 33.33, 55.56, 77.78, 100.0, 122.22, 144.44, 166.67, 188.89 }
    //strength := [9]float64{ 0.74, 1.83, 2.92, 4.01, 5.1, 6.19, 7.28, 8.37, 9.46 }
    fmt.Println(length[i_length])


    count_state_arr := []int{10, 20, 30, 40}
    save_to_file(strings.Trim(strings.Join(strings.Fields(fmt.Sprint(count_state_arr)), ","), "[]"))





}