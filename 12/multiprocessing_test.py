import multiprocessing
import time
import math


def calc_distance(pi, threads, threads_arr, result_dis_workers, result_ids_workers):
    #while not q_x.empty():
    #    print(pi, q_x.get())
    #time.sleep(1)
    #if not q_x.empty():
    #    print(pi, q_x.get())
    #else:
    #    time.sleep(1)
    for i in range(pi*8, len(threads_arr), 6*8):
        tr = [threads_arr[i] for i in range(i, i+8)]
        print(pi, tr)
        dis = min([math.sqrt(sum(map(lambda x: (x[0] - x[1])**2, zip(thread, tr)))) for thread in threads])
        #time.sleep(1)



if __name__ == "__main__":

    threads = {
        0: [[2,5,6,7,2,5,1,9],[3,9,2,8,2,6,1,6],[4,9,7,3,3,3,3,9],[5,7,9,6,1,4,2,6],[6,5,4,3,3,2,1,2],[7,6,5,4,4,3,2,2],[9,6,1,7,1,2,3,9]],
        1: [[0,0,0,0,1,1,1,0],[2,0,7,4,5,7,7,7],[4,0,3,6,3,4,9,9],[5,8,2,0,5,8,5,3],[6,4,8,8,4,2,7,7],[7,5,0,6,9,1,3,6],[8,4,1,5,3,4,8,3]]
    }




    #q_x = multiprocessing.Queue()

    threads_arr = multiprocessing.Array('i', 48*3)
    #start_workers = multiprocessing.Value('i', 1)

    num_proc = 6

    result_dis_workers = multiprocessing.Array('d', num_proc)
    result_ids_workers = multiprocessing.Array('i', num_proc)

    def_dis_val = 1e10
    for i in range(num_proc):
        result_dis_workers[i] = def_dis_val

    for i in range(48*3):
        threads_arr[i] = i


    #for i in range(3000):
    #    q_x.put(i)

    ps = []
    for pi in range(num_proc):
        ps.append(multiprocessing.Process(target=calc_distance, args=(pi, threads[0], threads_arr, result_dis_workers, result_ids_workers)))

    for p in ps:
        p.start()


    #for p in ps:
    #    print(p.is_alive())

    
    #for i in range(99, 10):
    #    q_x.put(i)
    #    time.sleep(1)
    

    #print('!')

    #start_workers.value = 0


    '''
    while 1:
        if q_x.empty():
            print('empty')
            for p in ps:
                p.terminate()
            break
        else:
            time.sleep(1)
    '''


    for p in ps:
        p.join()

    print('End')


    '''
    p1 = multiprocessing.Process(target=square_list, args=(mylist, result, square_sum))
    p1.start()
    p1.join()
    print(result[:])
    print(square_sum.value)
    '''


    #q_x.put([1,2,3,4])
    #print(q_x.get(), q_x.empty(), start_workers.value, result_dis_workers[:])







