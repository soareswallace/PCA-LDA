import numpy as np
from csv import reader
import random# Load a CSV file

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def data_balance(dataset):
    dataset_copy = []
    cont = 0
    (quantity1, st_class) = ((False == dataset[:, -1]).sum(), 0) #encontrei a quantidade de dados de uma classe
    (quantity2, nd_class) = (len(dataset) - st_class, 1) #e da outra
    if quantity1 < quantity2:
        for row in range(len(dataset)):
            if (dataset[row][-1] == st_class) and (cont is not quantity1):
                dataset_copy.append(dataset[row])
                cont += 1
            else:
                dataset_copy.append(dataset[row])
                cont += 1
            if(cont == (2*quantity1)):
                return dataset_copy

    else:
        for row in range(len(dataset)):
            if (dataset[row][-1] == st_class) and (cont is not quantity2):
                dataset_copy.append(dataset[row])
                cont += 1
            else:
                dataset_copy.append(dataset[row])
                cont += 1
            if (cont == (2 * quantity2)):
                return dataset_copy


def split_data(df, split, lvq_training_set, knn_test_set):
    random.shuffle(df[0])
    for i in range(len(df) -1):
        if i < split:
            lvq_training_set.append(df[i])
        else:
            knn_test_set.append(df[i])

def data_conversion_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    cleanup_nums = {"defect": {"true": 4, "false": 2}}
    # pego apenas o ultimo valor, ou seja a classe.
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup