package main

import (
	"fmt"
	"math"
	"sync"
)

func (s *State) hypot(p1 []float64, p2 []float64) float64{
	return math.Sqrt(math.Pow(p1[0]-p2[0], 2) + math.Pow(p1[1]-p2[1], 2) + math.Pow(p1[2]-p2[2], 2) + math.Pow(p1[3]-p2[3], 2) + math.Pow(p1[4]-p2[4], 2) + math.Pow(p1[5]-p2[5], 2) + math.Pow(p1[6]-p2[6], 2) + math.Pow(p1[7]-p2[7], 2))
}

func (s *State) calc_distance(a1 int, point []float64) {
	s.tmp_dis[a1] = 999.0
	for i := 0; i < len(s.zero); i++ {
		s.tmp_dis[a1] = math.Min(s.hypot(s.zero[i], point), s.tmp_dis[a1])
		//fmt.Println(a1, "..0")
	}
	for i := 0; i < len(s.one); i++ {
		if s.tmp_dis[a1] > s.hypot(s.one[i], point) {
			//fmt.Println(a1, "..1")
			return
		}
	}
	if s.tmp_dis[a1] > s.result_dis[a1] {
		s.result_dis[a1] = s.tmp_dis[a1]
		s.result_points[a1] = point
	}

	fmt.Println(s.tmp_dis[a1], point)
}

func (s *State) go_calc(a1 int) {
	fmt.Println(a1)
	point := []float64{float64(a1), 1.0, 2.0, 3.0, 5.0, 5.0, 6.0, 7.0}
	s.calc_distance(a1, point)
	point = []float64{float64(a1), 1.0, 5.0, 3.0, 7.0, 5.0, 6.0, 7.0}
	s.calc_distance(a1, point)
	point = []float64{float64(a1), 3.0, 3.0, 3.0, 6.0, 9.0, 9.0, 8.0}
	s.calc_distance(a1, point)
	s.wg.Done()
}



type State struct {
    zero, one [][]float64
    tmp_dis [10]float64
    result_dis [10]float64
    result_points [10][]float64
    max_dis float64
    max_point []float64
    n [10]float64
    wg sync.WaitGroup
}

func init_state() *State {
    s := new(State)
    s.n = [10]float64{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
    return s
}

func main() {

	s := init_state()

	s.zero = [][]float64{
		{0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0},
		{1.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0},
		{2.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0},
	}
	s.one = [][]float64{
		{9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0},
		{9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 8.0},
		{9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 7.0},
	}

	//a0 := 0
	//n := [10]float64{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	//point := [8]float64{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}



	s.wg.Add(1)
	go s.go_calc(0)
	s.wg.Add(1)
	go s.go_calc(1)
	s.wg.Add(1)
	go s.go_calc(2)
	s.wg.Wait()

	s.max_dis = 0
	for i := 0; i < len(s.result_dis); i++ {
		if s.result_dis[i] > s.max_dis {
			s.max_dis = s.result_dis[i]
			s.max_point = s.result_points[i]
		}
	}

	fmt.Println(s.max_dis, s.max_point)
	fmt.Println("End")


	/*
	for a1 := 0; a1 < len(n); a1++ {
		for a2 := 0; a2 < len(n); a2++ {
			for a3 := 0; a3 < len(n); a3++ {
				for a4 := 0; a4 < len(n); a4++ {
					for a5 := 0; a5 < len(n); a5++ {
						for a6 := 0; a6 < len(n); a6++ {
							for a7 := 0; a7 < len(n); a7++ {
								if (a4 != 0 && a5 != 0 && a6 != 0) {
									point := [8]float64{n[a0], n[a1], n[a2], n[a3], n[a4], n[a5], n[a6], n[a7]}
									fmt.Println(point)
								}
							}
						}
					}
				}
			}
		}
	}*/

	//c := make(chan int, 10)
	//go fibonacci(cap(c), c)
	//for i := range c {
	//	fmt.Println(i)
	//}
}




