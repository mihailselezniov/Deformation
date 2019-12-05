package main

import (
    "fmt"
    "math"
    "sync"
    "time"
    "bufio"
    "os"
    "strings"
    "strconv"
)


func (s *State) hypot(p1 []float64, p2 []float64) float64{
    return math.Pow(p1[0]-p2[0], 2) + math.Pow(p1[1]-p2[1], 2) + math.Pow(p1[2]-p2[2], 2) + math.Pow(p1[3]-p2[3], 2) + math.Pow(p1[4]-p2[4], 2) + math.Pow(p1[5]-p2[5], 2) + math.Pow(p1[6]-p2[6], 2) + math.Pow(p1[7]-p2[7], 2)
}

func (s *State) calc_distance(index_point int, a0 int, point []float64) {
    s.tmp_dis[a0] = 99999.0
    for i := 0; i < s.len_zero; i++ {
        s.tmp_dis[a0] = math.Min(s.hypot(s.zero[i], point), s.tmp_dis[a0])
    }
    if s.tmp_dis[a0] == s.result_dis[a0] {
        s.tmp_dis1[a0] = 99999.0
        for i := 0; i < s.len_one; i++ {
            s.tmp_dis1[a0] = math.Min(s.hypot(s.one[i], point), s.tmp_dis1[a0])
        }
        if s.tmp_dis1[a0] > s.result_dis1[a0] {
            s.result_dis[a0] = s.tmp_dis[a0]
            s.result_dis1[a0] = s.tmp_dis1[a0]
            s.result_i_point[a0] = index_point
        }
    }
    if s.tmp_dis[a0] > s.result_dis[a0] {
        s.tmp_dis1[a0] = 99999.0
        for i := 0; i < s.len_one; i++ {
            s.tmp_dis1[a0] = math.Min(s.hypot(s.one[i], point), s.tmp_dis1[a0])
        }
        s.result_dis[a0] = s.tmp_dis[a0]
        s.result_dis1[a0] = s.tmp_dis1[a0]
        s.result_i_point[a0] = index_point
    }
}

func (s *State) load_y(a0 int) {
    file_data_is_broken, err := os.Open(s.f_y[a0])
    if err != nil {fmt.Println(err)}
    defer file_data_is_broken.Close()
    scanner_data_is_broken := bufio.NewScanner(file_data_is_broken)
    s.i_broken[a0] = 0
    s.i_y[a0] = 0
    for scanner_data_is_broken.Scan() {
        row := scanner_data_is_broken.Text()
        count_num, err := strconv.Atoi(row)
        if err != nil {fmt.Println(err)}
        s.val_bool[a0] = true
        if s.i_broken[a0]%2 == 0 {
            s.val_bool[a0] = false
        }
        for i := 0; i < count_num; i++ {
            s.y[a0][s.i_y[a0]] = s.val_bool[a0]
            s.i_y[a0] += 1
        }
        s.i_broken[a0] += 1
    }
    //fmt.Println(len(s.y[a0]))
}

func (s *State) go_calc(a0 int) {
    s.load_y(a0)
    index_point := a0 * s.Y_step
    index_point_y := 0
    for a1 := 0; a1 < s.len_n; a1++ {
        for a2 := 0; a2 < s.len_n; a2++ {
            for a3 := 0; a3 < s.len_n; a3++ {
                for a4 := 0; a4 < s.len_n; a4++ {
                    for a5 := 0; a5 < s.len_n; a5++ {
                        for a6 := 0; a6 < s.len_n; a6++ {
                            for a7 := 0; a7 < s.len_n; a7++ {
                                if (a4 != 0 && a5 != 0 && a6 != 0) {
                                    //fmt.Println(a0, index_point_y)
                                    if s.y[a0][index_point_y] == false {
                                        point := []float64{s.n[a0], s.n[a1], s.n[a2], s.n[a3], s.n[a4], s.n[a5], s.n[a6], s.n[a7]}
                                        //point := []float64{s.n[a0], s.n[a2], s.n[a3], s.n[a4], s.n[a5], s.n[a6], s.n[a7]}
                                        s.calc_distance(index_point, a0, point)
                                    }
                                    index_point += 1
                                    index_point_y += 1
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    s.wg.Done()
}

func (s *State) goroutines_calc() {
    for a0 := 0; a0 < s.len_n; a0++ {
        s.wg.Add(1)
        go s.go_calc(a0)
        //s.go_calc(a0)
    }
    s.wg.Wait()
}

func (s *State) calc_max_dis() {
    s.max_dis = -1
    s.max_dis1 = -1
    for i := 0; i < len(s.result_dis); i++ {
        if s.result_dis[i] == s.max_dis {
            if s.result_dis1[i] > s.max_dis1 {
                s.max_dis = s.result_dis[i]
                s.max_dis1 = s.result_dis1[i]
                s.max_i_point = s.result_i_point[i]
            }
        }
        if s.result_dis[i] > s.max_dis {
            s.max_dis = s.result_dis[i]
            s.max_dis1 = s.result_dis1[i]
            s.max_i_point = s.result_i_point[i]
        }
    }
    fmt.Println(s.max_dis, s.max_dis1, s.max_i_point)
}

func (s *State) load_threads() {
    // Open file with start points
    file, err := os.Open(s.f_threads)
    if err != nil {fmt.Println(err)}
    defer file.Close()

    // Read file
    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        row := scanner.Text()
        arr := strings.Split(row, ",")
        //fmt.Println(arr[8])
        var arr_point [9]float64
        for i := 0; i < len(arr); i++ {
            arr_point[i], err = strconv.ParseFloat(arr[i], 64)
            if err != nil {fmt.Println(err)}
        }
        if arr_point[8] == 0.0 {
            s.zero = append(s.zero, arr_point[0:8])
        }
        if arr_point[8] == 1.0 {
            s.one = append(s.one, arr_point[0:8])
        }
    }
    //fmt.Println(s.zero, s.one)
    s.len_zero, s.len_one = len(s.zero), len(s.one)
}

type State struct {
    zero, one [][]float64
    tmp_dis [10]float64
    tmp_dis1 [10]float64
    result_dis [10]float64
    result_dis1 [10]float64
    result_i_point [10]int
    max_dis, max_dis1 float64
    max_i_point int
    n [10]float64
    y [10][7290000]bool
    val_bool [10]bool
    i_y, i_broken [10]int
    len_n, len_zero, len_one int
    f_threads string
    ex_name string
    f_y [10]string
    Y_step int
    wg sync.WaitGroup
}

func init_state() *State {
    s := new(State)
    s.n = [10]float64{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
    s.len_n = len(s.n)
    s.result_dis = [10]float64{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
    s.ex_name = "6_1.txt"
    s.f_threads = "ml_threads/" + s.ex_name
    s.Y_step = 7290000
    for i := 0; i < len(s.f_y); i++ {
        s.f_y[i] = "ml_y_pred/y" + strconv.Itoa(i) + "_" + s.ex_name
    }
    return s
}

func main() {
    start_time := time.Now()
    s := init_state()
    s.load_threads()
    s.goroutines_calc()
    s.calc_max_dis()
    fmt.Println("#", s.len_zero + s.len_one, "(", s.len_zero, s.len_one, ")", time.Since(start_time))
}




