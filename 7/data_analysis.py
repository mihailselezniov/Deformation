import csv
from fiber_test_master import params

param_names = sorted(list(params.keys())+['is_broken'])
#print(param_names)

def get_data():
    with open('data2.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        for row in csv_reader:
            obj_row = {}
            for i in range(len(param_names)):
                if param_names[i] == 'diameter':
                    obj_row[param_names[i]] = int(float(row[i])*100)
                else:
                    obj_row[param_names[i]] = int(row[i])
            data.append(obj_row)
        return data

if __name__ == '__main__':
    data = get_data()
    #print(data[0])

    # Count points for each combination
    # How many times did not break the fiber
    combinations = {}
    for d in data:
        key = ','.join(map(str, [d['density'], d['diameter'], d['strength'], d['young']]))
        if key not in combinations:
            combinations[key] = []
        combinations[key].append(1 if not int(d['is_broken']) else 0)

    for i in sorted([[sum(combinations[c]), c] for c in combinations])[::-1]:
        print(i)
    # the greater 'density' and 'diameter' the better
