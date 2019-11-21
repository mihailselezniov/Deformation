import multiprocessing
import gc



def calc_distance(pi, threads, threads_arr, result_dis_workers, result_ids_workers):
    for l0 in threads_arr:
        dis = min([math.sqrt(sum(map(lambda x: (x[0] - x[1])**2, zip(thread, l0)))) for thread in threads])
        if dis > result_dis_workers[pi]:
            result_dis_workers[pi] = dis
            result_ids_workers[pi] = int(''.join(map(str, l0)))



if __name__ == "__main__":

    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.externals import joblib
    import collections
    import math
    import sys
    import time


    ml_type = int(sys.argv[1])

    '''
    threads = {
        0: [[2,5,6,7,2,5,1,9],[3,9,2,8,2,6,1,6],[4,9,7,3,3,3,3,9],[5,7,9,6,1,4,2,6],[6,5,4,3,3,2,1,2],[7,6,5,4,4,3,2,2],[9,6,1,7,1,2,3,9]],
        1: [[0,0,0,0,1,1,1,0],[2,0,7,4,5,7,7,7],[4,0,3,6,3,4,9,9],[5,8,2,0,5,8,5,3],[6,4,8,8,4,2,7,7],[7,5,0,6,9,1,3,6],[8,4,1,5,3,4,8,3]]
    }
    joblib.dump(threads, 'threads_{}.j'.format(ml_type))
    '''
    threads_f = 'threads/threads_{}'.format(ml_type)
    threads_f_name = '{}.j'.format(threads_f)
    threads72_f_name = '{}_72.j'.format(threads_f)
    threads729_f_name = '{}_729.j'.format(threads_f)
    threads = joblib.load(threads_f_name)
    len_threads = len(threads[0]) + len(threads[1])
    print('threads', len_threads)
    print(sys.getsizeof(threads))
    gc.collect()

    way_dict = joblib.load('border_vars/way_dict.j')
    print('way_dict')
    gc.collect()
    X = joblib.load('border_vars/X.j')
    print('X')
    gc.collect()
    Y = joblib.load('border_vars/Y.j')
    print('Y')
    gc.collect()

    print(sys.getsizeof(way_dict))
    print(X.shape, Y.shape)
    print('all', dict(collections.Counter(Y)))
    print(sys.getsizeof(X), sys.getsizeof(Y))

'''
    # import warnings filter
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)


    x_train, y_train = [], []
    model = ''
    if ml_type == 1:
        model = XGBClassifier()
    elif ml_type == 2:
        model = LogisticRegression()
    elif ml_type == 3:
        model = LinearSVC(random_state=0, tol=1e-5)
    elif ml_type == 4:
        model = KNeighborsClassifier(n_neighbors=6)

    def gen_train():
        global x_train
        global y_train
        x_train, y_train = [], []
        for i in threads:
            for thread in threads[i]:
                x_train.append(thread)
                y_train.append(i)
        x_train, y_train = np.array(x_train), np.array(y_train)

    while len_threads < 729:
        print('#', len_threads)
        gen_train()

        model.fit(x_train, y_train)
        y_pred = model.predict(X)
        accuracy = accuracy_score(Y, y_pred)
        print('Accuracy: {}'.format(accuracy))
        #print(y_pred)
        print(dict(collections.Counter(y_pred)))

        find = 1 if len(threads[0]) > len(threads[1]) else 0

        num_proc = 6
        threads_arr = {}
        for i in range(num_proc):
            threads_arr[i] = []


        pi_id = 0
        for i in range(len(y_pred)):
            if y_pred[i] == 0:
                for j in way_dict[i]:
                    if y_pred[j] == 1:
                        t_id = j if find else i
                        threads_arr[pi_id%num_proc].append(list(X[t_id]))
                        pi_id += 1


        result_dis_workers = multiprocessing.Array('d', num_proc)
        result_ids_workers = multiprocessing.Array('i', num_proc)

        for i in range(num_proc):
            result_dis_workers[i] = 0

        ps = []
        for pi in range(num_proc):
            ps.append(multiprocessing.Process(target=calc_distance, args=(pi, threads[0], threads_arr[pi], result_dis_workers, result_ids_workers)))

        for p in ps:
            p.start()

        for p in ps:
            p.join()

        tr = sorted(list(zip(result_dis_workers[:], result_ids_workers[:])), reverse=True)[0]
        print(tr)
        max_dis_thread = tr[1]

        #print([max_distance, max_dis_thread])
        print(Y[max_dis_thread])
        threads[Y[max_dis_thread]].append(list(X[max_dis_thread]))
        len_threads = len(threads[0]) + len(threads[1])
        if len_threads == 72:
            joblib.dump(threads, threads72_f_name)
        if len_threads == 729:
            joblib.dump(threads, threads729_f_name)
        joblib.dump(threads, threads_f_name)
        #print(threads)
        #break
'''




'''
(72900000, 8) (72900000,)
all {1: 63518691, 0: 9381309}
# 14
Accuracy: 0.6998067352537722
{1: 43837200, 0: 29062800}
[14.2828568570857, 5102990]
1
# 15
Accuracy: 0.8173903017832648
{1: 55836000, 0: 17064000}
[13.74772708486752, 728920]
1
# 16
Accuracy: 0.8739810562414266
{1: 61618500, 0: 11281500}
[12.806248474865697, 6568210]
1
# 17
Accuracy: 0.7952048148148149
{1: 59262840, 0: 13637160}
[13.674794331177344, 6563420]
1
# 18
Accuracy: 0.8727472290809328
{1: 61503690, 0: 11396310}
[12.083045973594572, 66196359]
1
# 19
Accuracy: 0.8599926886145405
{1: 60287000, 0: 12613000}
[12.569805089976535, 4737775]
1
# 20
Accuracy: 0.8837192181069958
{1: 64671600, 0: 8228400}
[11.958260743101398, 14507180]
1
# 21
Accuracy: 0.8887181481481482
{1: 65633496, 0: 7266504}
[12.0, 65902329]
1
# 22
Accuracy: 0.901887146776406
{1: 66510792, 0: 6389208}
[11.489125293076057, 7289200]
1
# 23
Accuracy: 0.8843528532235939
{1: 67988760, 0: 4911240}
[10.816653826391969, 72173348]
1
# 24
Accuracy: 0.8747271742112482
{1: 67849660, 0: 5050340}
[11.224972160321824, 65890]
1
# 25
Accuracy: 0.8746583676268861
{1: 65884988, 0: 7015012}
[11.489125293076057, 19019780]
1
# 26
Accuracy: 0.900230329218107
{1: 67433814, 0: 5466186}
[10.583005244258363, 71951949]
0
# 27
Accuracy: 0.9030360356652949
{1: 67006394, 0: 5893606}
[11.090536506409418, 71448669]
0
# 28
Accuracy: 0.8997987791495199
{1: 67240612, 0: 5659388}
[12.041594578792296, 66272769]
1
# 29
Accuracy: 0.8939306310013717
{1: 66757940, 0: 6142060}
[11.74734012447073, 26381969]
1
# 30
Accuracy: 0.9056758984910837
{1: 66176598, 0: 6723402}
[11.357816691600547, 6560641]
0
# 31
Accuracy: 0.9058410013717421
{1: 65053670, 0: 7846330}
[11.832159566199232, 4599905]
1
# 32
Accuracy: 0.9102788614540467
{1: 66562280, 0: 6337720}
[11.045361017187261, 6582511]
1
# 33
Accuracy: 0.8981539780521262
{1: 66075660, 0: 6824340}
[10.816653826391969, 65026710]
1
'''








