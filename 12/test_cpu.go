package main

import (  
    "fmt"
    "runtime"
    "time"
    "sync"
    "math/rand"
)

type G struct {
    wg sync.WaitGroup
    n [10]int
}

func (g *G) go_calc(in int) {
    x := 0
    for i := 0; i < 10000000; i++ {
        x += rand.Intn(2)
    }
    g.n[in] = x
    //g.wg.Done()
}

func main() {
    fmt.Println(runtime.NumCPU())
    g := new(G)
    start_time := time.Now()
    for i := 0; i < 10; i++ {
        //g.wg.Add(1)
        g.go_calc(i)
    }
    //g.wg.Wait()
    fmt.Println("End", time.Since(start_time))
}