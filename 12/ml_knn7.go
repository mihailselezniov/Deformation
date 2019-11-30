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

func (s *State) calc_distance(index_point int, a1 int, point []float64) {
    s.tmp_dis[a1] = 99999.0
    for i := 0; i < s.len_zero; i++ {
        s.tmp_dis[a1] = math.Min(s.hypot(s.zero[i], point), s.tmp_dis[a1])
        //fmt.Println(a1, "..0", s.hypot(s.zero[i], point), point)
    }
    for i := 0; i < s.len_one; i++ {
        if s.tmp_dis[a1] > s.hypot(s.one[i], point) {
            //fmt.Println(a1, "..1")
            return
        }
    }
    if s.tmp_dis[a1] > s.result_dis[a1] {
        s.result_dis[a1] = s.tmp_dis[a1]
        s.result_points[a1] = point
        s.result_i_point[a1] = index_point
    }

    //fmt.Println(s.tmp_dis[a1], point)
}

func (s *State) go_calc(a0 int) {
    //fmt.Println(a0)
    //start_goroutine_time := time.Now()

    index_point := a0 * 10000000
    for a1 := 0; a1 < s.len_n; a1++ {
        for a2 := 0; a2 < s.len_n; a2++ {
            for a3 := 0; a3 < s.len_n; a3++ {
                for a4 := 0; a4 < s.len_n; a4++ {
                    for a5 := 0; a5 < s.len_n; a5++ {
                        for a6 := 0; a6 < s.len_n; a6++ {
                            for a7 := 0; a7 < s.len_n; a7++ {
                                if (a4 != 0 && a5 != 0 && a6 != 0) {
                                    point := []float64{s.n[a0], s.n[a1], s.n[a2], s.n[a3], s.n[a4], s.n[a5], s.n[a6], s.n[a7]}
                                    s.calc_distance(index_point, a0, point)
                                }
                                index_point += 1
                            }
                        }
                    }
                }
            }
        }
    }
    //fmt.Println("index_point", a0, index_point)
    //fmt.Println("End goroutine", time.Since(start_goroutine_time))
    s.wg.Done()
}

func (s *State) goroutines_calc() {
    for a0 := 0; a0 < s.len_n; a0++ {
        s.wg.Add(1)
        go s.go_calc(a0)
    }
    s.wg.Wait()
}

func (s *State) load_y() {
    f_name_broken := "../11/fib_all_data.txt"
    file_data_is_broken, err := os.Open(f_name_broken)
    if err != nil {fmt.Println(err)}
    defer file_data_is_broken.Close()
    scanner_data_is_broken := bufio.NewScanner(file_data_is_broken)
    s.i_broken = 0
    s.i_y = 0
    for scanner_data_is_broken.Scan() {
        row := scanner_data_is_broken.Text()
        count_num, err := strconv.Atoi(row)
        if err != nil {fmt.Println(err)}
        s.val_bool = true
        if s.i_broken%2 == 0 {
            s.val_bool = false
        }
        for i := 0; i < count_num; i++ {
            s.y[s.i_y] = s.val_bool
            s.i_y += 1
        }
        s.i_broken += 1
    }
    //fmt.Println(s.i_y)
}

func (s *State) load_threads(f_name string) {
    //f_name := "ml_threads/4_1.txt"
    // 4_1 => KNN1 + Squared Euclidean distance
    // 4_2 => KNN1 + Manhattan distance
    // 4_3 => KNN1 + Maximum norm
    // 4_4 => KNN1 + Cosine similarity

    // Open file with start points
    file, err := os.Open(f_name)
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

func (s *State) calc_max_dis() {
    s.max_dis = -1
    for i := 0; i < len(s.result_dis); i++ {
        if s.result_dis[i] > s.max_dis {
            s.max_dis = s.result_dis[i]
            s.max_point = s.result_points[i]
            s.max_i_point = s.result_i_point[i]
        }
    }
    fmt.Println(s.max_dis, s.max_i_point, s.max_point, s.y[s.max_i_point])
}

func (s *State) make_row_thread() {
    s.row_thread = ""
    for i := 0; i < len(s.max_point); i++ {
        s.row_thread += strconv.Itoa(int(s.max_point[i]))
        s.row_thread += ","
    }
    if s.y[s.max_i_point] {
        s.row_thread += "1"
    } else {
        s.row_thread += "0"
    }
    s.row_thread += "\n"
    //fmt.Println(s.row_thread)
}

func (s *State) save_local_thread() {
    if s.y[s.max_i_point] {
        s.one = append(s.one, s.max_point[:])
    } else {
        s.zero = append(s.zero, s.max_point[:])
    }
    s.len_zero, s.len_one = len(s.zero), len(s.one)
    //fmt.Println(s.len_zero, s.len_one, s.one, s.zero)
}

func (s *State) save_thread_to_file() {
    f, err := os.OpenFile(s.f_threads, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if err != nil { fmt.Println(err) }
    if _, err := f.Write([]byte(s.row_thread)); err != nil { fmt.Println(err) }
    if err := f.Close(); err != nil { fmt.Println(err) }
}

func (s *State) clean_state() {
    s.result_dis = [10]float64{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
}

type State struct {
    zero, one [][]float64
    tmp_dis [10]float64
    result_dis [10]float64
    result_points [10][]float64
    result_i_point [10]int
    max_dis float64
    max_point []float64
    max_i_point int
    n [10]float64
    y [100000000]bool
    val_bool bool
    len_n, len_zero, len_one, i_y, i_broken int
    f_threads, row_thread string
    wg sync.WaitGroup
}

func init_state() *State {
    s := new(State)
    s.n = [10]float64{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
    s.len_n = len(s.n)
    s.result_dis = [10]float64{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
    s.f_threads = "ml_threads/4_1.txt"
    return s
}

func main() {

    start_time := time.Now()
    s := init_state()

    s.load_y()
    s.load_threads(s.f_threads)

    for {
        fmt.Println("#", s.len_zero + s.len_one, "(", s.len_zero, s.len_one, ")", time.Since(start_time))
        s.goroutines_calc()
        s.calc_max_dis()
        //s.max_i_point = 3579389
        //s.max_point = []float64{0, 4, 9, 0, 9, 9, 9, 9}
        s.save_local_thread()
        s.make_row_thread()
        s.save_thread_to_file()
        s.clean_state()
    }
    


    fmt.Println("End", time.Since(start_time))

}




