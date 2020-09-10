# -*- coding: utf-8 -*-


opt_activations = ['tanh', 'relu']
opt_solvers = ['adam']
opt_hidden_layer_sizes = [(30, 30), (50, 50), (70, 70), (70, 30), (100, 10), (30, 70), (10, 100), (30, 50, 70), (70, 50, 30), (10, 50, 100), (100, 50, 10), (30, 70, 30), (70, 30, 70), (10, 100, 10), (100, 10, 100), (30, 30, 30, 30), (50, 50, 50, 50), (70, 70, 70, 70), (30, 30, 30, 30, 30), (50, 50, 50, 50, 50), (70, 70, 70, 70, 70)][::-1]


def make_rating(metrics):
    data = []
    i = 0
    for inx_activation, opt_activation in enumerate(opt_activations):
        for inx_solver, opt_solver in enumerate(opt_solvers):
            for inx, layer in enumerate(opt_hidden_layer_sizes):
                data.append({'name':'{} + {}'.format(opt_activation, opt_solver), 'val': metrics[i][inx], 'layer': opt_hidden_layer_sizes[inx]})
            i += 1

    #print(data)
    #data_sort = sorted(data, key=lambda x: x['val'], reverse=True)
    #for i in data_sort:
    #    print(i)
    return data

def sort_data(data):
    data_sort = sorted(data, key=lambda x: x['val'], reverse=True)
    for i in data_sort:
        print(i)

roc_metrics = [[0.9192, 0.9047, 0.9191, 0.9147, 0.8938, 0.8919, 0.8877, 0.8873, 0.9146, 0.899, 0.8948, 0.8341, 0.9053, 0.9132, 0.8962, 0.8909, 0.8906, 0.917, 0.9102, 0.9063, 0.9031], [0.9153, 0.9004, 0.9016, 0.9105, 0.9099, 0.9006, 0.9096, 0.9017, 0.9043, 0.9077, 0.906, 0.8642, 0.9145, 0.9055, 0.8766, 0.9042, 0.9062, 0.9182, 0.905, 0.9158, 0.8957]]
pr_metrics = [[0.9561, 0.9479, 0.9563, 0.9537, 0.9421, 0.9411, 0.939, 0.9393, 0.9536, 0.9449, 0.9427, 0.9125, 0.9483, 0.9528, 0.9439, 0.9406, 0.9409, 0.9553, 0.9511, 0.9489, 0.9473], [0.9552, 0.9464, 0.947, 0.9528, 0.9518, 0.9461, 0.9514, 0.9498, 0.9481, 0.95, 0.9491, 0.9327, 0.9545, 0.9491, 0.9364, 0.9479, 0.9488, 0.9563, 0.9486, 0.9542, 0.9433]]
f1_metrics = [[0.8819, 0.8762, 0.8771, 0.8762, 0.8628, 0.8617, 0.8558, 0.8428, 0.8785, 0.8684, 0.8627, 0.768, 0.8736, 0.8777, 0.853, 0.8616, 0.848, 0.8721, 0.8748, 0.8734, 0.8681], [0.8544, 0.8528, 0.8536, 0.8448, 0.8568, 0.8613, 0.8627, 0.8076, 0.864, 0.8663, 0.8652, 0.7258, 0.8589, 0.8583, 0.7766, 0.8693, 0.8749, 0.8667, 0.8641, 0.8789, 0.8621]]

data_roc_metrics = make_rating(roc_metrics)
data_pr_metrics = make_rating(pr_metrics)
data_f1_metrics = make_rating(f1_metrics)



opt_activations = ['identity', 'logistic', 'tanh', 'relu']
opt_solvers = ['lbfgs', 'sgd', 'adam']
opt_hidden_layer_sizes = [(i,i,i) for i in range(10, 200, 20)]

roc_metrics = [[0.8246, 0.8246, 0.8246, 0.8245, 0.8246, 0.8247, 0.8246, 0.8244, 0.8246, 0.8245], [0.8241, 0.859, 0.8588, 0.8448, 0.8643, 0.8644, 0.8636, 0.8639, 0.8602, 0.8665], [0.8352, 0.8353, 0.8263, 0.8302, 0.8366, 0.8449, 0.8287, 0.8355, 0.8372, 0.8307], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.9023, 0.9035, 0.8885, 0.902, 0.8973, 0.8954, 0.8946, 0.9098, 0.8914, 0.8987], [0.5074, 0.5, 0.5157, 0.5, 0.5, 0.7282, 0.8636, 0.5006, 0.8643, 0.6969], [0.8964, 0.8947, 0.8974, 0.9146, 0.9116, 0.9042, 0.8317, 0.9133, 0.8865, 0.8902], [0.8769, 0.8847, 0.9015, 0.8885, 0.8729, 0.8776, 0.913, 0.8928, 0.9034, 0.9108], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.8848, 0.9012, 0.9216, 0.91, 0.9109, 0.9061, 0.9107, 0.9141, 0.9096, 0.9195]]
pr_metrics = [[0.9078, 0.9078, 0.9078, 0.9078, 0.9078, 0.9078, 0.9078, 0.9077, 0.9078, 0.9078], [0.9172, 0.9265, 0.9264, 0.9254, 0.93, 0.9315, 0.9295, 0.9314, 0.9272, 0.9322], [0.9132, 0.9133, 0.9087, 0.9107, 0.914, 0.9184, 0.9099, 0.9134, 0.9143, 0.9109], [0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736], [0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736], [0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736], [0.9486, 0.9485, 0.9399, 0.9472, 0.9446, 0.9436, 0.9433, 0.9518, 0.9413, 0.9455], [0.777, 0.7736, 0.7807, 0.7736, 0.7736, 0.876, 0.9296, 0.7738, 0.9309, 0.8625], [0.9434, 0.9425, 0.9441, 0.9534, 0.9517, 0.9477, 0.9113, 0.9529, 0.9382, 0.9403], [0.9348, 0.9376, 0.9468, 0.9402, 0.9313, 0.9339, 0.953, 0.9427, 0.9481, 0.9523], [0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736, 0.7736], [0.9399, 0.9468, 0.9578, 0.9515, 0.9521, 0.9493, 0.9521, 0.9536, 0.9513, 0.9569]]
f1_metrics = [[0.7567, 0.7567, 0.7567, 0.7567, 0.7567, 0.7568, 0.7567, 0.7566, 0.7568, 0.7567], [0.6381, 0.7682, 0.7665, 0.6775, 0.7618, 0.7404, 0.7614, 0.7384, 0.7684, 0.7489], [0.7655, 0.7649, 0.7569, 0.7598, 0.7645, 0.7686, 0.7601, 0.7647, 0.7643, 0.7617], [0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692], [0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692], [0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692], [0.8309, 0.8459, 0.844, 0.8563, 0.8525, 0.851, 0.847, 0.8556, 0.8493, 0.8511], [0.3727, 0.3692, 0.3767, 0.3692, 0.3692, 0.5195, 0.7611, 0.3695, 0.748, 0.4914], [0.8679, 0.8673, 0.8648, 0.8834, 0.8812, 0.8718, 0.7637, 0.8762, 0.8578, 0.8581], [0.8098, 0.8459, 0.8568, 0.8371, 0.839, 0.8373, 0.8691, 0.8353, 0.8532, 0.8579], [0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692, 0.3692], [0.7995, 0.8545, 0.8766, 0.8655, 0.8641, 0.8613, 0.8609, 0.8709, 0.8645, 0.8711]]

data_roc_metrics += make_rating(roc_metrics)
data_pr_metrics += make_rating(pr_metrics)
data_f1_metrics += make_rating(f1_metrics)

sort_data(data_roc_metrics)
print()
sort_data(data_pr_metrics)
print()
sort_data(data_f1_metrics)


"""
{'name': 'relu + adam', 'val': 0.9216, 'layer': (50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9195, 'layer': (190, 190, 190)}
{'name': 'tanh + adam', 'val': 0.9192, 'layer': (70, 70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9191, 'layer': (30, 30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.9182, 'layer': (70, 30)}
{'name': 'tanh + adam', 'val': 0.917, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.9158, 'layer': (50, 50)}
{'name': 'relu + adam', 'val': 0.9153, 'layer': (70, 70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9147, 'layer': (70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9146, 'layer': (70, 30, 70)}
{'name': 'tanh + adam', 'val': 0.9146, 'layer': (70, 70, 70)}
{'name': 'relu + adam', 'val': 0.9145, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.9141, 'layer': (150, 150, 150)}
{'name': 'tanh + adam', 'val': 0.9133, 'layer': (150, 150, 150)}
{'name': 'tanh + adam', 'val': 0.9132, 'layer': (30, 50, 70)}
{'name': 'relu + lbfgs', 'val': 0.913, 'layer': (130, 130, 130)}
{'name': 'tanh + adam', 'val': 0.9116, 'layer': (90, 90, 90)}
{'name': 'relu + adam', 'val': 0.9109, 'layer': (90, 90, 90)}
{'name': 'relu + lbfgs', 'val': 0.9108, 'layer': (190, 190, 190)}
{'name': 'relu + adam', 'val': 0.9107, 'layer': (130, 130, 130)}
{'name': 'relu + adam', 'val': 0.9105, 'layer': (70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9102, 'layer': (70, 70)}
{'name': 'relu + adam', 'val': 0.91, 'layer': (70, 70, 70)}
{'name': 'relu + adam', 'val': 0.9099, 'layer': (50, 50, 50, 50)}
{'name': 'tanh + lbfgs', 'val': 0.9098, 'layer': (150, 150, 150)}
{'name': 'relu + adam', 'val': 0.9096, 'layer': (100, 10, 100)}
{'name': 'relu + adam', 'val': 0.9096, 'layer': (170, 170, 170)}
{'name': 'relu + adam', 'val': 0.9077, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.9063, 'layer': (50, 50)}
{'name': 'relu + adam', 'val': 0.9062, 'layer': (100, 10)}
{'name': 'relu + adam', 'val': 0.9061, 'layer': (110, 110, 110)}
{'name': 'relu + adam', 'val': 0.906, 'layer': (100, 50, 10)}
{'name': 'relu + adam', 'val': 0.9055, 'layer': (30, 50, 70)}
{'name': 'tanh + adam', 'val': 0.9053, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.905, 'layer': (70, 70)}
{'name': 'tanh + adam', 'val': 0.9047, 'layer': (50, 50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9043, 'layer': (70, 30, 70)}
{'name': 'relu + adam', 'val': 0.9042, 'layer': (30, 70)}
{'name': 'tanh + adam', 'val': 0.9042, 'layer': (110, 110, 110)}
{'name': 'tanh + lbfgs', 'val': 0.9035, 'layer': (30, 30, 30)}
{'name': 'relu + lbfgs', 'val': 0.9034, 'layer': (170, 170, 170)}
{'name': 'tanh + adam', 'val': 0.9031, 'layer': (30, 30)}
{'name': 'tanh + lbfgs', 'val': 0.9023, 'layer': (10, 10, 10)}
{'name': 'tanh + lbfgs', 'val': 0.902, 'layer': (70, 70, 70)}
{'name': 'relu + adam', 'val': 0.9017, 'layer': (10, 100, 10)}
{'name': 'relu + adam', 'val': 0.9016, 'layer': (30, 30, 30, 30, 30)}
{'name': 'relu + lbfgs', 'val': 0.9015, 'layer': (50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9012, 'layer': (30, 30, 30)}
{'name': 'relu + adam', 'val': 0.9006, 'layer': (30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.9004, 'layer': (50, 50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.899, 'layer': (30, 70, 30)}
{'name': 'tanh + lbfgs', 'val': 0.8987, 'layer': (190, 190, 190)}
{'name': 'tanh + adam', 'val': 0.8974, 'layer': (50, 50, 50)}
{'name': 'tanh + lbfgs', 'val': 0.8973, 'layer': (90, 90, 90)}
{'name': 'tanh + adam', 'val': 0.8964, 'layer': (10, 10, 10)}
{'name': 'tanh + adam', 'val': 0.8962, 'layer': (10, 100)}
{'name': 'relu + adam', 'val': 0.8957, 'layer': (30, 30)}
{'name': 'tanh + lbfgs', 'val': 0.8954, 'layer': (110, 110, 110)}
{'name': 'tanh + adam', 'val': 0.8948, 'layer': (100, 50, 10)}
{'name': 'tanh + adam', 'val': 0.8947, 'layer': (30, 30, 30)}
{'name': 'tanh + lbfgs', 'val': 0.8946, 'layer': (130, 130, 130)}
{'name': 'tanh + adam', 'val': 0.8938, 'layer': (50, 50, 50, 50)}
{'name': 'relu + lbfgs', 'val': 0.8928, 'layer': (150, 150, 150)}
{'name': 'tanh + adam', 'val': 0.8919, 'layer': (30, 30, 30, 30)}
{'name': 'tanh + lbfgs', 'val': 0.8914, 'layer': (170, 170, 170)}
{'name': 'tanh + adam', 'val': 0.8909, 'layer': (30, 70)}
{'name': 'tanh + adam', 'val': 0.8906, 'layer': (100, 10)}
{'name': 'tanh + adam', 'val': 0.8902, 'layer': (190, 190, 190)}
{'name': 'tanh + lbfgs', 'val': 0.8885, 'layer': (50, 50, 50)}
{'name': 'relu + lbfgs', 'val': 0.8885, 'layer': (70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.8877, 'layer': (100, 10, 100)}
{'name': 'tanh + adam', 'val': 0.8873, 'layer': (10, 100, 10)}
{'name': 'tanh + adam', 'val': 0.8865, 'layer': (170, 170, 170)}
{'name': 'relu + adam', 'val': 0.8848, 'layer': (10, 10, 10)}
{'name': 'relu + lbfgs', 'val': 0.8847, 'layer': (30, 30, 30)}
{'name': 'relu + lbfgs', 'val': 0.8776, 'layer': (110, 110, 110)}
{'name': 'relu + lbfgs', 'val': 0.8769, 'layer': (10, 10, 10)}
{'name': 'relu + adam', 'val': 0.8766, 'layer': (10, 100)}
{'name': 'relu + lbfgs', 'val': 0.8729, 'layer': (90, 90, 90)}
{'name': 'identity + sgd', 'val': 0.8665, 'layer': (190, 190, 190)}
{'name': 'identity + sgd', 'val': 0.8644, 'layer': (110, 110, 110)}
{'name': 'identity + sgd', 'val': 0.8643, 'layer': (90, 90, 90)}
{'name': 'tanh + sgd', 'val': 0.8643, 'layer': (170, 170, 170)}
{'name': 'relu + adam', 'val': 0.8642, 'layer': (10, 50, 100)}
{'name': 'identity + sgd', 'val': 0.8639, 'layer': (150, 150, 150)}
{'name': 'identity + sgd', 'val': 0.8636, 'layer': (130, 130, 130)}
{'name': 'tanh + sgd', 'val': 0.8636, 'layer': (130, 130, 130)}
{'name': 'identity + sgd', 'val': 0.8602, 'layer': (170, 170, 170)}
{'name': 'identity + sgd', 'val': 0.859, 'layer': (30, 30, 30)}
{'name': 'identity + sgd', 'val': 0.8588, 'layer': (50, 50, 50)}
{'name': 'identity + adam', 'val': 0.8449, 'layer': (110, 110, 110)}
{'name': 'identity + sgd', 'val': 0.8448, 'layer': (70, 70, 70)}
{'name': 'identity + adam', 'val': 0.8372, 'layer': (170, 170, 170)}
{'name': 'identity + adam', 'val': 0.8366, 'layer': (90, 90, 90)}
{'name': 'identity + adam', 'val': 0.8355, 'layer': (150, 150, 150)}
{'name': 'identity + adam', 'val': 0.8353, 'layer': (30, 30, 30)}
{'name': 'identity + adam', 'val': 0.8352, 'layer': (10, 10, 10)}
{'name': 'tanh + adam', 'val': 0.8341, 'layer': (10, 50, 100)}
{'name': 'tanh + adam', 'val': 0.8317, 'layer': (130, 130, 130)}
{'name': 'identity + adam', 'val': 0.8307, 'layer': (190, 190, 190)}
{'name': 'identity + adam', 'val': 0.8302, 'layer': (70, 70, 70)}
{'name': 'identity + adam', 'val': 0.8287, 'layer': (130, 130, 130)}
{'name': 'identity + adam', 'val': 0.8263, 'layer': (50, 50, 50)}
{'name': 'identity + lbfgs', 'val': 0.8247, 'layer': (110, 110, 110)}
{'name': 'identity + lbfgs', 'val': 0.8246, 'layer': (10, 10, 10)}
{'name': 'identity + lbfgs', 'val': 0.8246, 'layer': (30, 30, 30)}
{'name': 'identity + lbfgs', 'val': 0.8246, 'layer': (50, 50, 50)}
{'name': 'identity + lbfgs', 'val': 0.8246, 'layer': (90, 90, 90)}
{'name': 'identity + lbfgs', 'val': 0.8246, 'layer': (130, 130, 130)}
{'name': 'identity + lbfgs', 'val': 0.8246, 'layer': (170, 170, 170)}
{'name': 'identity + lbfgs', 'val': 0.8245, 'layer': (70, 70, 70)}
{'name': 'identity + lbfgs', 'val': 0.8245, 'layer': (190, 190, 190)}
{'name': 'identity + lbfgs', 'val': 0.8244, 'layer': (150, 150, 150)}
{'name': 'identity + sgd', 'val': 0.8241, 'layer': (10, 10, 10)}
{'name': 'tanh + sgd', 'val': 0.7282, 'layer': (110, 110, 110)}
{'name': 'tanh + sgd', 'val': 0.6969, 'layer': (190, 190, 190)}
{'name': 'tanh + sgd', 'val': 0.5157, 'layer': (50, 50, 50)}
{'name': 'tanh + sgd', 'val': 0.5074, 'layer': (10, 10, 10)}
{'name': 'tanh + sgd', 'val': 0.5006, 'layer': (150, 150, 150)}
{'name': 'logistic + lbfgs', 'val': 0.5, 'layer': (10, 10, 10)}
{'name': 'logistic + lbfgs', 'val': 0.5, 'layer': (30, 30, 30)}
{'name': 'logistic + lbfgs', 'val': 0.5, 'layer': (50, 50, 50)}
{'name': 'logistic + lbfgs', 'val': 0.5, 'layer': (70, 70, 70)}
{'name': 'logistic + lbfgs', 'val': 0.5, 'layer': (90, 90, 90)}
{'name': 'logistic + lbfgs', 'val': 0.5, 'layer': (110, 110, 110)}
{'name': 'logistic + lbfgs', 'val': 0.5, 'layer': (130, 130, 130)}
{'name': 'logistic + lbfgs', 'val': 0.5, 'layer': (150, 150, 150)}
{'name': 'logistic + lbfgs', 'val': 0.5, 'layer': (170, 170, 170)}
{'name': 'logistic + lbfgs', 'val': 0.5, 'layer': (190, 190, 190)}
{'name': 'logistic + sgd', 'val': 0.5, 'layer': (10, 10, 10)}
{'name': 'logistic + sgd', 'val': 0.5, 'layer': (30, 30, 30)}
{'name': 'logistic + sgd', 'val': 0.5, 'layer': (50, 50, 50)}
{'name': 'logistic + sgd', 'val': 0.5, 'layer': (70, 70, 70)}
{'name': 'logistic + sgd', 'val': 0.5, 'layer': (90, 90, 90)}
{'name': 'logistic + sgd', 'val': 0.5, 'layer': (110, 110, 110)}
{'name': 'logistic + sgd', 'val': 0.5, 'layer': (130, 130, 130)}
{'name': 'logistic + sgd', 'val': 0.5, 'layer': (150, 150, 150)}
{'name': 'logistic + sgd', 'val': 0.5, 'layer': (170, 170, 170)}
{'name': 'logistic + sgd', 'val': 0.5, 'layer': (190, 190, 190)}
{'name': 'logistic + adam', 'val': 0.5, 'layer': (10, 10, 10)}
{'name': 'logistic + adam', 'val': 0.5, 'layer': (30, 30, 30)}
{'name': 'logistic + adam', 'val': 0.5, 'layer': (50, 50, 50)}
{'name': 'logistic + adam', 'val': 0.5, 'layer': (70, 70, 70)}
{'name': 'logistic + adam', 'val': 0.5, 'layer': (90, 90, 90)}
{'name': 'logistic + adam', 'val': 0.5, 'layer': (110, 110, 110)}
{'name': 'logistic + adam', 'val': 0.5, 'layer': (130, 130, 130)}
{'name': 'logistic + adam', 'val': 0.5, 'layer': (150, 150, 150)}
{'name': 'logistic + adam', 'val': 0.5, 'layer': (170, 170, 170)}
{'name': 'logistic + adam', 'val': 0.5, 'layer': (190, 190, 190)}
{'name': 'tanh + sgd', 'val': 0.5, 'layer': (30, 30, 30)}
{'name': 'tanh + sgd', 'val': 0.5, 'layer': (70, 70, 70)}
{'name': 'tanh + sgd', 'val': 0.5, 'layer': (90, 90, 90)}
{'name': 'relu + sgd', 'val': 0.5, 'layer': (10, 10, 10)}
{'name': 'relu + sgd', 'val': 0.5, 'layer': (30, 30, 30)}
{'name': 'relu + sgd', 'val': 0.5, 'layer': (50, 50, 50)}
{'name': 'relu + sgd', 'val': 0.5, 'layer': (70, 70, 70)}
{'name': 'relu + sgd', 'val': 0.5, 'layer': (90, 90, 90)}
{'name': 'relu + sgd', 'val': 0.5, 'layer': (110, 110, 110)}
{'name': 'relu + sgd', 'val': 0.5, 'layer': (130, 130, 130)}
{'name': 'relu + sgd', 'val': 0.5, 'layer': (150, 150, 150)}
{'name': 'relu + sgd', 'val': 0.5, 'layer': (170, 170, 170)}
{'name': 'relu + sgd', 'val': 0.5, 'layer': (190, 190, 190)}

{'name': 'relu + adam', 'val': 0.9578, 'layer': (50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9569, 'layer': (190, 190, 190)}
{'name': 'tanh + adam', 'val': 0.9563, 'layer': (30, 30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.9563, 'layer': (70, 30)}
{'name': 'tanh + adam', 'val': 0.9561, 'layer': (70, 70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9553, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.9552, 'layer': (70, 70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.9545, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.9542, 'layer': (50, 50)}
{'name': 'tanh + adam', 'val': 0.9537, 'layer': (70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9536, 'layer': (70, 30, 70)}
{'name': 'relu + adam', 'val': 0.9536, 'layer': (150, 150, 150)}
{'name': 'tanh + adam', 'val': 0.9534, 'layer': (70, 70, 70)}
{'name': 'relu + lbfgs', 'val': 0.953, 'layer': (130, 130, 130)}
{'name': 'tanh + adam', 'val': 0.9529, 'layer': (150, 150, 150)}
{'name': 'tanh + adam', 'val': 0.9528, 'layer': (30, 50, 70)}
{'name': 'relu + adam', 'val': 0.9528, 'layer': (70, 70, 70, 70)}
{'name': 'relu + lbfgs', 'val': 0.9523, 'layer': (190, 190, 190)}
{'name': 'relu + adam', 'val': 0.9521, 'layer': (90, 90, 90)}
{'name': 'relu + adam', 'val': 0.9521, 'layer': (130, 130, 130)}
{'name': 'relu + adam', 'val': 0.9518, 'layer': (50, 50, 50, 50)}
{'name': 'tanh + lbfgs', 'val': 0.9518, 'layer': (150, 150, 150)}
{'name': 'tanh + adam', 'val': 0.9517, 'layer': (90, 90, 90)}
{'name': 'relu + adam', 'val': 0.9515, 'layer': (70, 70, 70)}
{'name': 'relu + adam', 'val': 0.9514, 'layer': (100, 10, 100)}
{'name': 'relu + adam', 'val': 0.9513, 'layer': (170, 170, 170)}
{'name': 'tanh + adam', 'val': 0.9511, 'layer': (70, 70)}
{'name': 'relu + adam', 'val': 0.95, 'layer': (30, 70, 30)}
{'name': 'relu + adam', 'val': 0.9498, 'layer': (10, 100, 10)}
{'name': 'relu + adam', 'val': 0.9493, 'layer': (110, 110, 110)}
{'name': 'relu + adam', 'val': 0.9491, 'layer': (100, 50, 10)}
{'name': 'relu + adam', 'val': 0.9491, 'layer': (30, 50, 70)}
{'name': 'tanh + adam', 'val': 0.9489, 'layer': (50, 50)}
{'name': 'relu + adam', 'val': 0.9488, 'layer': (100, 10)}
{'name': 'relu + adam', 'val': 0.9486, 'layer': (70, 70)}
{'name': 'tanh + lbfgs', 'val': 0.9486, 'layer': (10, 10, 10)}
{'name': 'tanh + lbfgs', 'val': 0.9485, 'layer': (30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.9483, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.9481, 'layer': (70, 30, 70)}
{'name': 'relu + lbfgs', 'val': 0.9481, 'layer': (170, 170, 170)}
{'name': 'tanh + adam', 'val': 0.9479, 'layer': (50, 50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9479, 'layer': (30, 70)}
{'name': 'tanh + adam', 'val': 0.9477, 'layer': (110, 110, 110)}
{'name': 'tanh + adam', 'val': 0.9473, 'layer': (30, 30)}
{'name': 'tanh + lbfgs', 'val': 0.9472, 'layer': (70, 70, 70)}
{'name': 'relu + adam', 'val': 0.947, 'layer': (30, 30, 30, 30, 30)}
{'name': 'relu + lbfgs', 'val': 0.9468, 'layer': (50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9468, 'layer': (30, 30, 30)}
{'name': 'relu + adam', 'val': 0.9464, 'layer': (50, 50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9461, 'layer': (30, 30, 30, 30)}
{'name': 'tanh + lbfgs', 'val': 0.9455, 'layer': (190, 190, 190)}
{'name': 'tanh + adam', 'val': 0.9449, 'layer': (30, 70, 30)}
{'name': 'tanh + lbfgs', 'val': 0.9446, 'layer': (90, 90, 90)}
{'name': 'tanh + adam', 'val': 0.9441, 'layer': (50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.9439, 'layer': (10, 100)}
{'name': 'tanh + lbfgs', 'val': 0.9436, 'layer': (110, 110, 110)}
{'name': 'tanh + adam', 'val': 0.9434, 'layer': (10, 10, 10)}
{'name': 'relu + adam', 'val': 0.9433, 'layer': (30, 30)}
{'name': 'tanh + lbfgs', 'val': 0.9433, 'layer': (130, 130, 130)}
{'name': 'tanh + adam', 'val': 0.9427, 'layer': (100, 50, 10)}
{'name': 'relu + lbfgs', 'val': 0.9427, 'layer': (150, 150, 150)}
{'name': 'tanh + adam', 'val': 0.9425, 'layer': (30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.9421, 'layer': (50, 50, 50, 50)}
{'name': 'tanh + lbfgs', 'val': 0.9413, 'layer': (170, 170, 170)}
{'name': 'tanh + adam', 'val': 0.9411, 'layer': (30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.9409, 'layer': (100, 10)}
{'name': 'tanh + adam', 'val': 0.9406, 'layer': (30, 70)}
{'name': 'tanh + adam', 'val': 0.9403, 'layer': (190, 190, 190)}
{'name': 'relu + lbfgs', 'val': 0.9402, 'layer': (70, 70, 70)}
{'name': 'tanh + lbfgs', 'val': 0.9399, 'layer': (50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9399, 'layer': (10, 10, 10)}
{'name': 'tanh + adam', 'val': 0.9393, 'layer': (10, 100, 10)}
{'name': 'tanh + adam', 'val': 0.939, 'layer': (100, 10, 100)}
{'name': 'tanh + adam', 'val': 0.9382, 'layer': (170, 170, 170)}
{'name': 'relu + lbfgs', 'val': 0.9376, 'layer': (30, 30, 30)}
{'name': 'relu + adam', 'val': 0.9364, 'layer': (10, 100)}
{'name': 'relu + lbfgs', 'val': 0.9348, 'layer': (10, 10, 10)}
{'name': 'relu + lbfgs', 'val': 0.9339, 'layer': (110, 110, 110)}
{'name': 'relu + adam', 'val': 0.9327, 'layer': (10, 50, 100)}
{'name': 'identity + sgd', 'val': 0.9322, 'layer': (190, 190, 190)}
{'name': 'identity + sgd', 'val': 0.9315, 'layer': (110, 110, 110)}
{'name': 'identity + sgd', 'val': 0.9314, 'layer': (150, 150, 150)}
{'name': 'relu + lbfgs', 'val': 0.9313, 'layer': (90, 90, 90)}
{'name': 'tanh + sgd', 'val': 0.9309, 'layer': (170, 170, 170)}
{'name': 'identity + sgd', 'val': 0.93, 'layer': (90, 90, 90)}
{'name': 'tanh + sgd', 'val': 0.9296, 'layer': (130, 130, 130)}
{'name': 'identity + sgd', 'val': 0.9295, 'layer': (130, 130, 130)}
{'name': 'identity + sgd', 'val': 0.9272, 'layer': (170, 170, 170)}
{'name': 'identity + sgd', 'val': 0.9265, 'layer': (30, 30, 30)}
{'name': 'identity + sgd', 'val': 0.9264, 'layer': (50, 50, 50)}
{'name': 'identity + sgd', 'val': 0.9254, 'layer': (70, 70, 70)}
{'name': 'identity + adam', 'val': 0.9184, 'layer': (110, 110, 110)}
{'name': 'identity + sgd', 'val': 0.9172, 'layer': (10, 10, 10)}
{'name': 'identity + adam', 'val': 0.9143, 'layer': (170, 170, 170)}
{'name': 'identity + adam', 'val': 0.914, 'layer': (90, 90, 90)}
{'name': 'identity + adam', 'val': 0.9134, 'layer': (150, 150, 150)}
{'name': 'identity + adam', 'val': 0.9133, 'layer': (30, 30, 30)}
{'name': 'identity + adam', 'val': 0.9132, 'layer': (10, 10, 10)}
{'name': 'tanh + adam', 'val': 0.9125, 'layer': (10, 50, 100)}
{'name': 'tanh + adam', 'val': 0.9113, 'layer': (130, 130, 130)}
{'name': 'identity + adam', 'val': 0.9109, 'layer': (190, 190, 190)}
{'name': 'identity + adam', 'val': 0.9107, 'layer': (70, 70, 70)}
{'name': 'identity + adam', 'val': 0.9099, 'layer': (130, 130, 130)}
{'name': 'identity + adam', 'val': 0.9087, 'layer': (50, 50, 50)}
{'name': 'identity + lbfgs', 'val': 0.9078, 'layer': (10, 10, 10)}
{'name': 'identity + lbfgs', 'val': 0.9078, 'layer': (30, 30, 30)}
{'name': 'identity + lbfgs', 'val': 0.9078, 'layer': (50, 50, 50)}
{'name': 'identity + lbfgs', 'val': 0.9078, 'layer': (70, 70, 70)}
{'name': 'identity + lbfgs', 'val': 0.9078, 'layer': (90, 90, 90)}
{'name': 'identity + lbfgs', 'val': 0.9078, 'layer': (110, 110, 110)}
{'name': 'identity + lbfgs', 'val': 0.9078, 'layer': (130, 130, 130)}
{'name': 'identity + lbfgs', 'val': 0.9078, 'layer': (170, 170, 170)}
{'name': 'identity + lbfgs', 'val': 0.9078, 'layer': (190, 190, 190)}
{'name': 'identity + lbfgs', 'val': 0.9077, 'layer': (150, 150, 150)}
{'name': 'tanh + sgd', 'val': 0.876, 'layer': (110, 110, 110)}
{'name': 'tanh + sgd', 'val': 0.8625, 'layer': (190, 190, 190)}
{'name': 'tanh + sgd', 'val': 0.7807, 'layer': (50, 50, 50)}
{'name': 'tanh + sgd', 'val': 0.777, 'layer': (10, 10, 10)}
{'name': 'tanh + sgd', 'val': 0.7738, 'layer': (150, 150, 150)}
{'name': 'logistic + lbfgs', 'val': 0.7736, 'layer': (10, 10, 10)}
{'name': 'logistic + lbfgs', 'val': 0.7736, 'layer': (30, 30, 30)}
{'name': 'logistic + lbfgs', 'val': 0.7736, 'layer': (50, 50, 50)}
{'name': 'logistic + lbfgs', 'val': 0.7736, 'layer': (70, 70, 70)}
{'name': 'logistic + lbfgs', 'val': 0.7736, 'layer': (90, 90, 90)}
{'name': 'logistic + lbfgs', 'val': 0.7736, 'layer': (110, 110, 110)}
{'name': 'logistic + lbfgs', 'val': 0.7736, 'layer': (130, 130, 130)}
{'name': 'logistic + lbfgs', 'val': 0.7736, 'layer': (150, 150, 150)}
{'name': 'logistic + lbfgs', 'val': 0.7736, 'layer': (170, 170, 170)}
{'name': 'logistic + lbfgs', 'val': 0.7736, 'layer': (190, 190, 190)}
{'name': 'logistic + sgd', 'val': 0.7736, 'layer': (10, 10, 10)}
{'name': 'logistic + sgd', 'val': 0.7736, 'layer': (30, 30, 30)}
{'name': 'logistic + sgd', 'val': 0.7736, 'layer': (50, 50, 50)}
{'name': 'logistic + sgd', 'val': 0.7736, 'layer': (70, 70, 70)}
{'name': 'logistic + sgd', 'val': 0.7736, 'layer': (90, 90, 90)}
{'name': 'logistic + sgd', 'val': 0.7736, 'layer': (110, 110, 110)}
{'name': 'logistic + sgd', 'val': 0.7736, 'layer': (130, 130, 130)}
{'name': 'logistic + sgd', 'val': 0.7736, 'layer': (150, 150, 150)}
{'name': 'logistic + sgd', 'val': 0.7736, 'layer': (170, 170, 170)}
{'name': 'logistic + sgd', 'val': 0.7736, 'layer': (190, 190, 190)}
{'name': 'logistic + adam', 'val': 0.7736, 'layer': (10, 10, 10)}
{'name': 'logistic + adam', 'val': 0.7736, 'layer': (30, 30, 30)}
{'name': 'logistic + adam', 'val': 0.7736, 'layer': (50, 50, 50)}
{'name': 'logistic + adam', 'val': 0.7736, 'layer': (70, 70, 70)}
{'name': 'logistic + adam', 'val': 0.7736, 'layer': (90, 90, 90)}
{'name': 'logistic + adam', 'val': 0.7736, 'layer': (110, 110, 110)}
{'name': 'logistic + adam', 'val': 0.7736, 'layer': (130, 130, 130)}
{'name': 'logistic + adam', 'val': 0.7736, 'layer': (150, 150, 150)}
{'name': 'logistic + adam', 'val': 0.7736, 'layer': (170, 170, 170)}
{'name': 'logistic + adam', 'val': 0.7736, 'layer': (190, 190, 190)}
{'name': 'tanh + sgd', 'val': 0.7736, 'layer': (30, 30, 30)}
{'name': 'tanh + sgd', 'val': 0.7736, 'layer': (70, 70, 70)}
{'name': 'tanh + sgd', 'val': 0.7736, 'layer': (90, 90, 90)}
{'name': 'relu + sgd', 'val': 0.7736, 'layer': (10, 10, 10)}
{'name': 'relu + sgd', 'val': 0.7736, 'layer': (30, 30, 30)}
{'name': 'relu + sgd', 'val': 0.7736, 'layer': (50, 50, 50)}
{'name': 'relu + sgd', 'val': 0.7736, 'layer': (70, 70, 70)}
{'name': 'relu + sgd', 'val': 0.7736, 'layer': (90, 90, 90)}
{'name': 'relu + sgd', 'val': 0.7736, 'layer': (110, 110, 110)}
{'name': 'relu + sgd', 'val': 0.7736, 'layer': (130, 130, 130)}
{'name': 'relu + sgd', 'val': 0.7736, 'layer': (150, 150, 150)}
{'name': 'relu + sgd', 'val': 0.7736, 'layer': (170, 170, 170)}
{'name': 'relu + sgd', 'val': 0.7736, 'layer': (190, 190, 190)}

{'name': 'tanh + adam', 'val': 0.8834, 'layer': (70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.8819, 'layer': (70, 70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.8812, 'layer': (90, 90, 90)}
{'name': 'relu + adam', 'val': 0.8789, 'layer': (50, 50)}
{'name': 'tanh + adam', 'val': 0.8785, 'layer': (70, 30, 70)}
{'name': 'tanh + adam', 'val': 0.8777, 'layer': (30, 50, 70)}
{'name': 'tanh + adam', 'val': 0.8771, 'layer': (30, 30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.8766, 'layer': (50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.8762, 'layer': (50, 50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.8762, 'layer': (70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.8762, 'layer': (150, 150, 150)}
{'name': 'relu + adam', 'val': 0.8749, 'layer': (100, 10)}
{'name': 'tanh + adam', 'val': 0.8748, 'layer': (70, 70)}
{'name': 'tanh + adam', 'val': 0.8736, 'layer': (70, 50, 30)}
{'name': 'tanh + adam', 'val': 0.8734, 'layer': (50, 50)}
{'name': 'tanh + adam', 'val': 0.8721, 'layer': (70, 30)}
{'name': 'tanh + adam', 'val': 0.8718, 'layer': (110, 110, 110)}
{'name': 'relu + adam', 'val': 0.8711, 'layer': (190, 190, 190)}
{'name': 'relu + adam', 'val': 0.8709, 'layer': (150, 150, 150)}
{'name': 'relu + adam', 'val': 0.8693, 'layer': (30, 70)}
{'name': 'relu + lbfgs', 'val': 0.8691, 'layer': (130, 130, 130)}
{'name': 'tanh + adam', 'val': 0.8684, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.8681, 'layer': (30, 30)}
{'name': 'tanh + adam', 'val': 0.8679, 'layer': (10, 10, 10)}
{'name': 'tanh + adam', 'val': 0.8673, 'layer': (30, 30, 30)}
{'name': 'relu + adam', 'val': 0.8667, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.8663, 'layer': (30, 70, 30)}
{'name': 'relu + adam', 'val': 0.8655, 'layer': (70, 70, 70)}
{'name': 'relu + adam', 'val': 0.8652, 'layer': (100, 50, 10)}
{'name': 'tanh + adam', 'val': 0.8648, 'layer': (50, 50, 50)}
{'name': 'relu + adam', 'val': 0.8645, 'layer': (170, 170, 170)}
{'name': 'relu + adam', 'val': 0.8641, 'layer': (70, 70)}
{'name': 'relu + adam', 'val': 0.8641, 'layer': (90, 90, 90)}
{'name': 'relu + adam', 'val': 0.864, 'layer': (70, 30, 70)}
{'name': 'tanh + adam', 'val': 0.8628, 'layer': (50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.8627, 'layer': (100, 50, 10)}
{'name': 'relu + adam', 'val': 0.8627, 'layer': (100, 10, 100)}
{'name': 'relu + adam', 'val': 0.8621, 'layer': (30, 30)}
{'name': 'tanh + adam', 'val': 0.8617, 'layer': (30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.8616, 'layer': (30, 70)}
{'name': 'relu + adam', 'val': 0.8613, 'layer': (30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.8613, 'layer': (110, 110, 110)}
{'name': 'relu + adam', 'val': 0.8609, 'layer': (130, 130, 130)}
{'name': 'relu + adam', 'val': 0.8589, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.8583, 'layer': (30, 50, 70)}
{'name': 'tanh + adam', 'val': 0.8581, 'layer': (190, 190, 190)}
{'name': 'relu + lbfgs', 'val': 0.8579, 'layer': (190, 190, 190)}
{'name': 'tanh + adam', 'val': 0.8578, 'layer': (170, 170, 170)}
{'name': 'relu + adam', 'val': 0.8568, 'layer': (50, 50, 50, 50)}
{'name': 'relu + lbfgs', 'val': 0.8568, 'layer': (50, 50, 50)}
{'name': 'tanh + lbfgs', 'val': 0.8563, 'layer': (70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.8558, 'layer': (100, 10, 100)}
{'name': 'tanh + lbfgs', 'val': 0.8556, 'layer': (150, 150, 150)}
{'name': 'relu + adam', 'val': 0.8545, 'layer': (30, 30, 30)}
{'name': 'relu + adam', 'val': 0.8544, 'layer': (70, 70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.8536, 'layer': (30, 30, 30, 30, 30)}
{'name': 'relu + lbfgs', 'val': 0.8532, 'layer': (170, 170, 170)}
{'name': 'tanh + adam', 'val': 0.853, 'layer': (10, 100)}
{'name': 'relu + adam', 'val': 0.8528, 'layer': (50, 50, 50, 50, 50)}
{'name': 'tanh + lbfgs', 'val': 0.8525, 'layer': (90, 90, 90)}
{'name': 'tanh + lbfgs', 'val': 0.8511, 'layer': (190, 190, 190)}
{'name': 'tanh + lbfgs', 'val': 0.851, 'layer': (110, 110, 110)}
{'name': 'tanh + lbfgs', 'val': 0.8493, 'layer': (170, 170, 170)}
{'name': 'tanh + adam', 'val': 0.848, 'layer': (100, 10)}
{'name': 'tanh + lbfgs', 'val': 0.847, 'layer': (130, 130, 130)}
{'name': 'tanh + lbfgs', 'val': 0.8459, 'layer': (30, 30, 30)}
{'name': 'relu + lbfgs', 'val': 0.8459, 'layer': (30, 30, 30)}
{'name': 'relu + adam', 'val': 0.8448, 'layer': (70, 70, 70, 70)}
{'name': 'tanh + lbfgs', 'val': 0.844, 'layer': (50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.8428, 'layer': (10, 100, 10)}
{'name': 'relu + lbfgs', 'val': 0.839, 'layer': (90, 90, 90)}
{'name': 'relu + lbfgs', 'val': 0.8373, 'layer': (110, 110, 110)}
{'name': 'relu + lbfgs', 'val': 0.8371, 'layer': (70, 70, 70)}
{'name': 'relu + lbfgs', 'val': 0.8353, 'layer': (150, 150, 150)}
{'name': 'tanh + lbfgs', 'val': 0.8309, 'layer': (10, 10, 10)}
{'name': 'relu + lbfgs', 'val': 0.8098, 'layer': (10, 10, 10)}
{'name': 'relu + adam', 'val': 0.8076, 'layer': (10, 100, 10)}
{'name': 'relu + adam', 'val': 0.7995, 'layer': (10, 10, 10)}
{'name': 'relu + adam', 'val': 0.7766, 'layer': (10, 100)}
{'name': 'identity + adam', 'val': 0.7686, 'layer': (110, 110, 110)}
{'name': 'identity + sgd', 'val': 0.7684, 'layer': (170, 170, 170)}
{'name': 'identity + sgd', 'val': 0.7682, 'layer': (30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.768, 'layer': (10, 50, 100)}
{'name': 'identity + sgd', 'val': 0.7665, 'layer': (50, 50, 50)}
{'name': 'identity + adam', 'val': 0.7655, 'layer': (10, 10, 10)}
{'name': 'identity + adam', 'val': 0.7649, 'layer': (30, 30, 30)}
{'name': 'identity + adam', 'val': 0.7647, 'layer': (150, 150, 150)}
{'name': 'identity + adam', 'val': 0.7645, 'layer': (90, 90, 90)}
{'name': 'identity + adam', 'val': 0.7643, 'layer': (170, 170, 170)}
{'name': 'tanh + adam', 'val': 0.7637, 'layer': (130, 130, 130)}
{'name': 'identity + sgd', 'val': 0.7618, 'layer': (90, 90, 90)}
{'name': 'identity + adam', 'val': 0.7617, 'layer': (190, 190, 190)}
{'name': 'identity + sgd', 'val': 0.7614, 'layer': (130, 130, 130)}
{'name': 'tanh + sgd', 'val': 0.7611, 'layer': (130, 130, 130)}
{'name': 'identity + adam', 'val': 0.7601, 'layer': (130, 130, 130)}
{'name': 'identity + adam', 'val': 0.7598, 'layer': (70, 70, 70)}
{'name': 'identity + adam', 'val': 0.7569, 'layer': (50, 50, 50)}
{'name': 'identity + lbfgs', 'val': 0.7568, 'layer': (110, 110, 110)}
{'name': 'identity + lbfgs', 'val': 0.7568, 'layer': (170, 170, 170)}
{'name': 'identity + lbfgs', 'val': 0.7567, 'layer': (10, 10, 10)}
{'name': 'identity + lbfgs', 'val': 0.7567, 'layer': (30, 30, 30)}
{'name': 'identity + lbfgs', 'val': 0.7567, 'layer': (50, 50, 50)}
{'name': 'identity + lbfgs', 'val': 0.7567, 'layer': (70, 70, 70)}
{'name': 'identity + lbfgs', 'val': 0.7567, 'layer': (90, 90, 90)}
{'name': 'identity + lbfgs', 'val': 0.7567, 'layer': (130, 130, 130)}
{'name': 'identity + lbfgs', 'val': 0.7567, 'layer': (190, 190, 190)}
{'name': 'identity + lbfgs', 'val': 0.7566, 'layer': (150, 150, 150)}
{'name': 'identity + sgd', 'val': 0.7489, 'layer': (190, 190, 190)}
{'name': 'tanh + sgd', 'val': 0.748, 'layer': (170, 170, 170)}
{'name': 'identity + sgd', 'val': 0.7404, 'layer': (110, 110, 110)}
{'name': 'identity + sgd', 'val': 0.7384, 'layer': (150, 150, 150)}
{'name': 'relu + adam', 'val': 0.7258, 'layer': (10, 50, 100)}
{'name': 'identity + sgd', 'val': 0.6775, 'layer': (70, 70, 70)}
{'name': 'identity + sgd', 'val': 0.6381, 'layer': (10, 10, 10)}
{'name': 'tanh + sgd', 'val': 0.5195, 'layer': (110, 110, 110)}
{'name': 'tanh + sgd', 'val': 0.4914, 'layer': (190, 190, 190)}
{'name': 'tanh + sgd', 'val': 0.3767, 'layer': (50, 50, 50)}
{'name': 'tanh + sgd', 'val': 0.3727, 'layer': (10, 10, 10)}
{'name': 'tanh + sgd', 'val': 0.3695, 'layer': (150, 150, 150)}
{'name': 'logistic + lbfgs', 'val': 0.3692, 'layer': (10, 10, 10)}
{'name': 'logistic + lbfgs', 'val': 0.3692, 'layer': (30, 30, 30)}
{'name': 'logistic + lbfgs', 'val': 0.3692, 'layer': (50, 50, 50)}
{'name': 'logistic + lbfgs', 'val': 0.3692, 'layer': (70, 70, 70)}
{'name': 'logistic + lbfgs', 'val': 0.3692, 'layer': (90, 90, 90)}
{'name': 'logistic + lbfgs', 'val': 0.3692, 'layer': (110, 110, 110)}
{'name': 'logistic + lbfgs', 'val': 0.3692, 'layer': (130, 130, 130)}
{'name': 'logistic + lbfgs', 'val': 0.3692, 'layer': (150, 150, 150)}
{'name': 'logistic + lbfgs', 'val': 0.3692, 'layer': (170, 170, 170)}
{'name': 'logistic + lbfgs', 'val': 0.3692, 'layer': (190, 190, 190)}
{'name': 'logistic + sgd', 'val': 0.3692, 'layer': (10, 10, 10)}
{'name': 'logistic + sgd', 'val': 0.3692, 'layer': (30, 30, 30)}
{'name': 'logistic + sgd', 'val': 0.3692, 'layer': (50, 50, 50)}
{'name': 'logistic + sgd', 'val': 0.3692, 'layer': (70, 70, 70)}
{'name': 'logistic + sgd', 'val': 0.3692, 'layer': (90, 90, 90)}
{'name': 'logistic + sgd', 'val': 0.3692, 'layer': (110, 110, 110)}
{'name': 'logistic + sgd', 'val': 0.3692, 'layer': (130, 130, 130)}
{'name': 'logistic + sgd', 'val': 0.3692, 'layer': (150, 150, 150)}
{'name': 'logistic + sgd', 'val': 0.3692, 'layer': (170, 170, 170)}
{'name': 'logistic + sgd', 'val': 0.3692, 'layer': (190, 190, 190)}
{'name': 'logistic + adam', 'val': 0.3692, 'layer': (10, 10, 10)}
{'name': 'logistic + adam', 'val': 0.3692, 'layer': (30, 30, 30)}
{'name': 'logistic + adam', 'val': 0.3692, 'layer': (50, 50, 50)}
{'name': 'logistic + adam', 'val': 0.3692, 'layer': (70, 70, 70)}
{'name': 'logistic + adam', 'val': 0.3692, 'layer': (90, 90, 90)}
{'name': 'logistic + adam', 'val': 0.3692, 'layer': (110, 110, 110)}
{'name': 'logistic + adam', 'val': 0.3692, 'layer': (130, 130, 130)}
{'name': 'logistic + adam', 'val': 0.3692, 'layer': (150, 150, 150)}
{'name': 'logistic + adam', 'val': 0.3692, 'layer': (170, 170, 170)}
{'name': 'logistic + adam', 'val': 0.3692, 'layer': (190, 190, 190)}
{'name': 'tanh + sgd', 'val': 0.3692, 'layer': (30, 30, 30)}
{'name': 'tanh + sgd', 'val': 0.3692, 'layer': (70, 70, 70)}
{'name': 'tanh + sgd', 'val': 0.3692, 'layer': (90, 90, 90)}
{'name': 'relu + sgd', 'val': 0.3692, 'layer': (10, 10, 10)}
{'name': 'relu + sgd', 'val': 0.3692, 'layer': (30, 30, 30)}
{'name': 'relu + sgd', 'val': 0.3692, 'layer': (50, 50, 50)}
{'name': 'relu + sgd', 'val': 0.3692, 'layer': (70, 70, 70)}
{'name': 'relu + sgd', 'val': 0.3692, 'layer': (90, 90, 90)}
{'name': 'relu + sgd', 'val': 0.3692, 'layer': (110, 110, 110)}
{'name': 'relu + sgd', 'val': 0.3692, 'layer': (130, 130, 130)}
{'name': 'relu + sgd', 'val': 0.3692, 'layer': (150, 150, 150)}
{'name': 'relu + sgd', 'val': 0.3692, 'layer': (170, 170, 170)}
{'name': 'relu + sgd', 'val': 0.3692, 'layer': (190, 190, 190)}
"""





