# -*- coding: utf-8 -*-


opt_activations = ['tanh', 'relu']
opt_solvers = ['adam']
max_hidden_layer_sizes = 100
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
    data_sort = sorted(data, key=lambda x: x['val'], reverse=True)
    for i in data_sort:
        print(i)


roc_metrics = [[0.9192, 0.9047, 0.9191, 0.9147, 0.8938, 0.8919, 0.8877, 0.8873, 0.9146, 0.899, 0.8948, 0.8341, 0.9053, 0.9132, 0.8962, 0.8909, 0.8906, 0.917, 0.9102, 0.9063, 0.9031], [0.9153, 0.9004, 0.9016, 0.9105, 0.9099, 0.9006, 0.9096, 0.9017, 0.9043, 0.9077, 0.906, 0.8642, 0.9145, 0.9055, 0.8766, 0.9042, 0.9062, 0.9182, 0.905, 0.9158, 0.8957]]
pr_metrics = [[0.9561, 0.9479, 0.9563, 0.9537, 0.9421, 0.9411, 0.939, 0.9393, 0.9536, 0.9449, 0.9427, 0.9125, 0.9483, 0.9528, 0.9439, 0.9406, 0.9409, 0.9553, 0.9511, 0.9489, 0.9473], [0.9552, 0.9464, 0.947, 0.9528, 0.9518, 0.9461, 0.9514, 0.9498, 0.9481, 0.95, 0.9491, 0.9327, 0.9545, 0.9491, 0.9364, 0.9479, 0.9488, 0.9563, 0.9486, 0.9542, 0.9433]]
f1_metrics = [[0.8819, 0.8762, 0.8771, 0.8762, 0.8628, 0.8617, 0.8558, 0.8428, 0.8785, 0.8684, 0.8627, 0.768, 0.8736, 0.8777, 0.853, 0.8616, 0.848, 0.8721, 0.8748, 0.8734, 0.8681], [0.8544, 0.8528, 0.8536, 0.8448, 0.8568, 0.8613, 0.8627, 0.8076, 0.864, 0.8663, 0.8652, 0.7258, 0.8589, 0.8583, 0.7766, 0.8693, 0.8749, 0.8667, 0.8641, 0.8789, 0.8621]]


make_rating(roc_metrics)
print()
make_rating(pr_metrics)
print()
make_rating(f1_metrics)


"""
{'name': 'tanh + adam', 'val': 0.9192, 'layer': (70, 70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9191, 'layer': (30, 30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.9182, 'layer': (70, 30)}
{'name': 'tanh + adam', 'val': 0.917, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.9158, 'layer': (50, 50)}
{'name': 'relu + adam', 'val': 0.9153, 'layer': (70, 70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9147, 'layer': (70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9146, 'layer': (70, 30, 70)}
{'name': 'relu + adam', 'val': 0.9145, 'layer': (70, 50, 30)}
{'name': 'tanh + adam', 'val': 0.9132, 'layer': (30, 50, 70)}
{'name': 'relu + adam', 'val': 0.9105, 'layer': (70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9102, 'layer': (70, 70)}
{'name': 'relu + adam', 'val': 0.9099, 'layer': (50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9096, 'layer': (100, 10, 100)}
{'name': 'relu + adam', 'val': 0.9077, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.9063, 'layer': (50, 50)}
{'name': 'relu + adam', 'val': 0.9062, 'layer': (100, 10)}
{'name': 'relu + adam', 'val': 0.906, 'layer': (100, 50, 10)}
{'name': 'relu + adam', 'val': 0.9055, 'layer': (30, 50, 70)}
{'name': 'tanh + adam', 'val': 0.9053, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.905, 'layer': (70, 70)}
{'name': 'tanh + adam', 'val': 0.9047, 'layer': (50, 50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9043, 'layer': (70, 30, 70)}
{'name': 'relu + adam', 'val': 0.9042, 'layer': (30, 70)}
{'name': 'tanh + adam', 'val': 0.9031, 'layer': (30, 30)}
{'name': 'relu + adam', 'val': 0.9017, 'layer': (10, 100, 10)}
{'name': 'relu + adam', 'val': 0.9016, 'layer': (30, 30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.9006, 'layer': (30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.9004, 'layer': (50, 50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.899, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.8962, 'layer': (10, 100)}
{'name': 'relu + adam', 'val': 0.8957, 'layer': (30, 30)}
{'name': 'tanh + adam', 'val': 0.8948, 'layer': (100, 50, 10)}
{'name': 'tanh + adam', 'val': 0.8938, 'layer': (50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.8919, 'layer': (30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.8909, 'layer': (30, 70)}
{'name': 'tanh + adam', 'val': 0.8906, 'layer': (100, 10)}
{'name': 'tanh + adam', 'val': 0.8877, 'layer': (100, 10, 100)}
{'name': 'tanh + adam', 'val': 0.8873, 'layer': (10, 100, 10)}
{'name': 'relu + adam', 'val': 0.8766, 'layer': (10, 100)}
{'name': 'relu + adam', 'val': 0.8642, 'layer': (10, 50, 100)}
{'name': 'tanh + adam', 'val': 0.8341, 'layer': (10, 50, 100)}

{'name': 'tanh + adam', 'val': 0.9563, 'layer': (30, 30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.9563, 'layer': (70, 30)}
{'name': 'tanh + adam', 'val': 0.9561, 'layer': (70, 70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9553, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.9552, 'layer': (70, 70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.9545, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.9542, 'layer': (50, 50)}
{'name': 'tanh + adam', 'val': 0.9537, 'layer': (70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.9536, 'layer': (70, 30, 70)}
{'name': 'tanh + adam', 'val': 0.9528, 'layer': (30, 50, 70)}
{'name': 'relu + adam', 'val': 0.9528, 'layer': (70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.9518, 'layer': (50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9514, 'layer': (100, 10, 100)}
{'name': 'tanh + adam', 'val': 0.9511, 'layer': (70, 70)}
{'name': 'relu + adam', 'val': 0.95, 'layer': (30, 70, 30)}
{'name': 'relu + adam', 'val': 0.9498, 'layer': (10, 100, 10)}
{'name': 'relu + adam', 'val': 0.9491, 'layer': (100, 50, 10)}
{'name': 'relu + adam', 'val': 0.9491, 'layer': (30, 50, 70)}
{'name': 'tanh + adam', 'val': 0.9489, 'layer': (50, 50)}
{'name': 'relu + adam', 'val': 0.9488, 'layer': (100, 10)}
{'name': 'relu + adam', 'val': 0.9486, 'layer': (70, 70)}
{'name': 'tanh + adam', 'val': 0.9483, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.9481, 'layer': (70, 30, 70)}
{'name': 'tanh + adam', 'val': 0.9479, 'layer': (50, 50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9479, 'layer': (30, 70)}
{'name': 'tanh + adam', 'val': 0.9473, 'layer': (30, 30)}
{'name': 'relu + adam', 'val': 0.947, 'layer': (30, 30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.9464, 'layer': (50, 50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9461, 'layer': (30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.9449, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.9439, 'layer': (10, 100)}
{'name': 'relu + adam', 'val': 0.9433, 'layer': (30, 30)}
{'name': 'tanh + adam', 'val': 0.9427, 'layer': (100, 50, 10)}
{'name': 'tanh + adam', 'val': 0.9421, 'layer': (50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.9411, 'layer': (30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.9409, 'layer': (100, 10)}
{'name': 'tanh + adam', 'val': 0.9406, 'layer': (30, 70)}
{'name': 'tanh + adam', 'val': 0.9393, 'layer': (10, 100, 10)}
{'name': 'tanh + adam', 'val': 0.939, 'layer': (100, 10, 100)}
{'name': 'relu + adam', 'val': 0.9364, 'layer': (10, 100)}
{'name': 'relu + adam', 'val': 0.9327, 'layer': (10, 50, 100)}
{'name': 'tanh + adam', 'val': 0.9125, 'layer': (10, 50, 100)}

{'name': 'tanh + adam', 'val': 0.8819, 'layer': (70, 70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.8789, 'layer': (50, 50)}
{'name': 'tanh + adam', 'val': 0.8785, 'layer': (70, 30, 70)}
{'name': 'tanh + adam', 'val': 0.8777, 'layer': (30, 50, 70)}
{'name': 'tanh + adam', 'val': 0.8771, 'layer': (30, 30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.8762, 'layer': (50, 50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.8762, 'layer': (70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.8749, 'layer': (100, 10)}
{'name': 'tanh + adam', 'val': 0.8748, 'layer': (70, 70)}
{'name': 'tanh + adam', 'val': 0.8736, 'layer': (70, 50, 30)}
{'name': 'tanh + adam', 'val': 0.8734, 'layer': (50, 50)}
{'name': 'tanh + adam', 'val': 0.8721, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.8693, 'layer': (30, 70)}
{'name': 'tanh + adam', 'val': 0.8684, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.8681, 'layer': (30, 30)}
{'name': 'relu + adam', 'val': 0.8667, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.8663, 'layer': (30, 70, 30)}
{'name': 'relu + adam', 'val': 0.8652, 'layer': (100, 50, 10)}
{'name': 'relu + adam', 'val': 0.8641, 'layer': (70, 70)}
{'name': 'relu + adam', 'val': 0.864, 'layer': (70, 30, 70)}
{'name': 'tanh + adam', 'val': 0.8628, 'layer': (50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.8627, 'layer': (100, 50, 10)}
{'name': 'relu + adam', 'val': 0.8627, 'layer': (100, 10, 100)}
{'name': 'relu + adam', 'val': 0.8621, 'layer': (30, 30)}
{'name': 'tanh + adam', 'val': 0.8617, 'layer': (30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.8616, 'layer': (30, 70)}
{'name': 'relu + adam', 'val': 0.8613, 'layer': (30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.8589, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.8583, 'layer': (30, 50, 70)}
{'name': 'relu + adam', 'val': 0.8568, 'layer': (50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.8558, 'layer': (100, 10, 100)}
{'name': 'relu + adam', 'val': 0.8544, 'layer': (70, 70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.8536, 'layer': (30, 30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.853, 'layer': (10, 100)}
{'name': 'relu + adam', 'val': 0.8528, 'layer': (50, 50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.848, 'layer': (100, 10)}
{'name': 'relu + adam', 'val': 0.8448, 'layer': (70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.8428, 'layer': (10, 100, 10)}
{'name': 'relu + adam', 'val': 0.8076, 'layer': (10, 100, 10)}
{'name': 'relu + adam', 'val': 0.7766, 'layer': (10, 100)}
{'name': 'tanh + adam', 'val': 0.768, 'layer': (10, 50, 100)}
{'name': 'relu + adam', 'val': 0.7258, 'layer': (10, 50, 100)}
"""





