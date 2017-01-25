import pickle

def read_file(filename):
    layer_data = []
    with open('data/' + filename, 'r') as f:
        for line in f:
            layer_data.append(line.rstrip().split(','))
    return layer_data


layer_data_10k = read_file('cluster10k.h5')
layer_data_1k = read_file('cluster1k.h5')
layer_data_5k = read_file('cluster5k.h5')

row_names = [
        'Overall Mean: ',
        'Overall STD : ',
        'PS Mean     : ',
        'PS STD      : ']

for i in range(len(layer_data_10k)):
    print 'Layer %i' % i
    for j in range(4):
        line = ' '.join([row_names[j], layer_data_1k[i][j],
            layer_data_5k[i][j],
            layer_data_10k[i][j]])
        print line
    print ''
