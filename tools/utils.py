import numpy as np

def load_mnist_csv(file_path):    
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    labels = data[:, 0].astype(int)
    pixels = data[:, 1:] / 255.0
    num_samples = labels.shape[0]
    one_hot_labels = np.zeros((num_samples, 10))
    one_hot_labels[np.arange(num_samples), labels] = 1.0
    return pixels, one_hot_labels

def array_to_cords(number_input):
    y = np.floor(number_input / 28)
    x = number_input - y * 28
    return (x,y)