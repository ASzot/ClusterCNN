import pickle

def read_file(filename):
    layer_data = []
    with open('data/' + filename, 'r') as f:
        for line in f:
            layer_data.append(line.rstrip().split(','))
    return layer_data


layer_data_10k = read_file('cluster10k.h5')

print layer_data_10k
