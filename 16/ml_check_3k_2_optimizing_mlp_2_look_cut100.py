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


roc_metrics = [[0.7807, 0.7632, 0.7664, 0.7865, 0.7644, 0.7792, 0.7715, 0.7645, 0.7836, 0.7762, 0.7835, 0.778, 0.7706, 0.7823, 0.7694, 0.7759, 0.7734, 0.7895, 0.782, 0.7766, 0.7762], [0.8185, 0.7943, 0.779, 0.8215, 0.8232, 0.7736, 0.7997, 0.8218, 0.8059, 0.7864, 0.8049, 0.8468, 0.8262, 0.7905, 0.7973, 0.8016, 0.8106, 0.8114, 0.8111, 0.7992, 0.7904]]
pr_metrics = [[0.8865, 0.8785, 0.8799, 0.8892, 0.879, 0.8858, 0.8823, 0.8791, 0.8879, 0.8844, 0.8878, 0.8854, 0.8818, 0.8872, 0.8814, 0.8843, 0.8832, 0.8906, 0.8871, 0.8846, 0.8844], [0.9046, 0.8929, 0.8857, 0.9063, 0.9068, 0.8833, 0.8954, 0.9069, 0.8984, 0.8893, 0.8979, 0.9194, 0.9082, 0.8912, 0.8949, 0.8964, 0.9006, 0.901, 0.9009, 0.8952, 0.8911]]
f1_metrics = [[0.7075, 0.6815, 0.6858, 0.7161, 0.6837, 0.7044, 0.6919, 0.6802, 0.7097, 0.7004, 0.7124, 0.698, 0.6929, 0.7107, 0.6864, 0.7008, 0.6946, 0.7204, 0.7095, 0.7023, 0.6998], [0.7542, 0.7258, 0.7039, 0.753, 0.7636, 0.6952, 0.7336, 0.7398, 0.7422, 0.7082, 0.7412, 0.7695, 0.7688, 0.7184, 0.712, 0.734, 0.7514, 0.7492, 0.7494, 0.7342, 0.7209]]

make_rating(roc_metrics)
print()
make_rating(pr_metrics)
print()
make_rating(f1_metrics)


"""
{'name': 'relu + adam', 'val': 0.8468, 'layer': (10, 50, 100)}
{'name': 'relu + adam', 'val': 0.8262, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.8232, 'layer': (50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.8218, 'layer': (10, 100, 10)}
{'name': 'relu + adam', 'val': 0.8215, 'layer': (70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.8185, 'layer': (70, 70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.8114, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.8111, 'layer': (70, 70)}
{'name': 'relu + adam', 'val': 0.8106, 'layer': (100, 10)}
{'name': 'relu + adam', 'val': 0.8059, 'layer': (70, 30, 70)}
{'name': 'relu + adam', 'val': 0.8049, 'layer': (100, 50, 10)}
{'name': 'relu + adam', 'val': 0.8016, 'layer': (30, 70)}
{'name': 'relu + adam', 'val': 0.7997, 'layer': (100, 10, 100)}
{'name': 'relu + adam', 'val': 0.7992, 'layer': (50, 50)}
{'name': 'relu + adam', 'val': 0.7973, 'layer': (10, 100)}
{'name': 'relu + adam', 'val': 0.7943, 'layer': (50, 50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.7905, 'layer': (30, 50, 70)}
{'name': 'relu + adam', 'val': 0.7904, 'layer': (30, 30)}
{'name': 'tanh + adam', 'val': 0.7895, 'layer': (70, 30)}
{'name': 'tanh + adam', 'val': 0.7865, 'layer': (70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.7864, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.7836, 'layer': (70, 30, 70)}
{'name': 'tanh + adam', 'val': 0.7835, 'layer': (100, 50, 10)}
{'name': 'tanh + adam', 'val': 0.7823, 'layer': (30, 50, 70)}
{'name': 'tanh + adam', 'val': 0.782, 'layer': (70, 70)}
{'name': 'tanh + adam', 'val': 0.7807, 'layer': (70, 70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.7792, 'layer': (30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.779, 'layer': (30, 30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.778, 'layer': (10, 50, 100)}
{'name': 'tanh + adam', 'val': 0.7766, 'layer': (50, 50)}
{'name': 'tanh + adam', 'val': 0.7762, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.7762, 'layer': (30, 30)}
{'name': 'tanh + adam', 'val': 0.7759, 'layer': (30, 70)}
{'name': 'relu + adam', 'val': 0.7736, 'layer': (30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.7734, 'layer': (100, 10)}
{'name': 'tanh + adam', 'val': 0.7715, 'layer': (100, 10, 100)}
{'name': 'tanh + adam', 'val': 0.7706, 'layer': (70, 50, 30)}
{'name': 'tanh + adam', 'val': 0.7694, 'layer': (10, 100)}
{'name': 'tanh + adam', 'val': 0.7664, 'layer': (30, 30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.7645, 'layer': (10, 100, 10)}
{'name': 'tanh + adam', 'val': 0.7644, 'layer': (50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.7632, 'layer': (50, 50, 50, 50, 50)}

{'name': 'relu + adam', 'val': 0.9194, 'layer': (10, 50, 100)}
{'name': 'relu + adam', 'val': 0.9082, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.9069, 'layer': (10, 100, 10)}
{'name': 'relu + adam', 'val': 0.9068, 'layer': (50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.9063, 'layer': (70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.9046, 'layer': (70, 70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.901, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.9009, 'layer': (70, 70)}
{'name': 'relu + adam', 'val': 0.9006, 'layer': (100, 10)}
{'name': 'relu + adam', 'val': 0.8984, 'layer': (70, 30, 70)}
{'name': 'relu + adam', 'val': 0.8979, 'layer': (100, 50, 10)}
{'name': 'relu + adam', 'val': 0.8964, 'layer': (30, 70)}
{'name': 'relu + adam', 'val': 0.8954, 'layer': (100, 10, 100)}
{'name': 'relu + adam', 'val': 0.8952, 'layer': (50, 50)}
{'name': 'relu + adam', 'val': 0.8949, 'layer': (10, 100)}
{'name': 'relu + adam', 'val': 0.8929, 'layer': (50, 50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.8912, 'layer': (30, 50, 70)}
{'name': 'relu + adam', 'val': 0.8911, 'layer': (30, 30)}
{'name': 'tanh + adam', 'val': 0.8906, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.8893, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.8892, 'layer': (70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.8879, 'layer': (70, 30, 70)}
{'name': 'tanh + adam', 'val': 0.8878, 'layer': (100, 50, 10)}
{'name': 'tanh + adam', 'val': 0.8872, 'layer': (30, 50, 70)}
{'name': 'tanh + adam', 'val': 0.8871, 'layer': (70, 70)}
{'name': 'tanh + adam', 'val': 0.8865, 'layer': (70, 70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.8858, 'layer': (30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.8857, 'layer': (30, 30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.8854, 'layer': (10, 50, 100)}
{'name': 'tanh + adam', 'val': 0.8846, 'layer': (50, 50)}
{'name': 'tanh + adam', 'val': 0.8844, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.8844, 'layer': (30, 30)}
{'name': 'tanh + adam', 'val': 0.8843, 'layer': (30, 70)}
{'name': 'relu + adam', 'val': 0.8833, 'layer': (30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.8832, 'layer': (100, 10)}
{'name': 'tanh + adam', 'val': 0.8823, 'layer': (100, 10, 100)}
{'name': 'tanh + adam', 'val': 0.8818, 'layer': (70, 50, 30)}
{'name': 'tanh + adam', 'val': 0.8814, 'layer': (10, 100)}
{'name': 'tanh + adam', 'val': 0.8799, 'layer': (30, 30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.8791, 'layer': (10, 100, 10)}
{'name': 'tanh + adam', 'val': 0.879, 'layer': (50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.8785, 'layer': (50, 50, 50, 50, 50)}

{'name': 'relu + adam', 'val': 0.7695, 'layer': (10, 50, 100)}
{'name': 'relu + adam', 'val': 0.7688, 'layer': (70, 50, 30)}
{'name': 'relu + adam', 'val': 0.7636, 'layer': (50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.7542, 'layer': (70, 70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.753, 'layer': (70, 70, 70, 70)}
{'name': 'relu + adam', 'val': 0.7514, 'layer': (100, 10)}
{'name': 'relu + adam', 'val': 0.7494, 'layer': (70, 70)}
{'name': 'relu + adam', 'val': 0.7492, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.7422, 'layer': (70, 30, 70)}
{'name': 'relu + adam', 'val': 0.7412, 'layer': (100, 50, 10)}
{'name': 'relu + adam', 'val': 0.7398, 'layer': (10, 100, 10)}
{'name': 'relu + adam', 'val': 0.7342, 'layer': (50, 50)}
{'name': 'relu + adam', 'val': 0.734, 'layer': (30, 70)}
{'name': 'relu + adam', 'val': 0.7336, 'layer': (100, 10, 100)}
{'name': 'relu + adam', 'val': 0.7258, 'layer': (50, 50, 50, 50, 50)}
{'name': 'relu + adam', 'val': 0.7209, 'layer': (30, 30)}
{'name': 'tanh + adam', 'val': 0.7204, 'layer': (70, 30)}
{'name': 'relu + adam', 'val': 0.7184, 'layer': (30, 50, 70)}
{'name': 'tanh + adam', 'val': 0.7161, 'layer': (70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.7124, 'layer': (100, 50, 10)}
{'name': 'relu + adam', 'val': 0.712, 'layer': (10, 100)}
{'name': 'tanh + adam', 'val': 0.7107, 'layer': (30, 50, 70)}
{'name': 'tanh + adam', 'val': 0.7097, 'layer': (70, 30, 70)}
{'name': 'tanh + adam', 'val': 0.7095, 'layer': (70, 70)}
{'name': 'relu + adam', 'val': 0.7082, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.7075, 'layer': (70, 70, 70, 70, 70)}
{'name': 'tanh + adam', 'val': 0.7044, 'layer': (30, 30, 30, 30)}
{'name': 'relu + adam', 'val': 0.7039, 'layer': (30, 30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.7023, 'layer': (50, 50)}
{'name': 'tanh + adam', 'val': 0.7008, 'layer': (30, 70)}
{'name': 'tanh + adam', 'val': 0.7004, 'layer': (30, 70, 30)}
{'name': 'tanh + adam', 'val': 0.6998, 'layer': (30, 30)}
{'name': 'tanh + adam', 'val': 0.698, 'layer': (10, 50, 100)}
{'name': 'relu + adam', 'val': 0.6952, 'layer': (30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.6946, 'layer': (100, 10)}
{'name': 'tanh + adam', 'val': 0.6929, 'layer': (70, 50, 30)}
{'name': 'tanh + adam', 'val': 0.6919, 'layer': (100, 10, 100)}
{'name': 'tanh + adam', 'val': 0.6864, 'layer': (10, 100)}
{'name': 'tanh + adam', 'val': 0.6858, 'layer': (30, 30, 30, 30, 30)}
{'name': 'tanh + adam', 'val': 0.6837, 'layer': (50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.6815, 'layer': (50, 50, 50, 50, 50)}
{'name': 'tanh + adam', 'val': 0.6802, 'layer': (10, 100, 10)}
"""





