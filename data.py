import csv
import random
import os
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

infile = open('file_name.csv', 'r')
outfile = open('output.csv', 'w', newline='')

reader = csv.reader(infile)
writer = csv.writer(outfile)

header = next(reader)

columns_to_keep = [0, 1]

new_header = [header[i] for i in columns_to_keep]

writer.writerow(new_header)

for row in reader:
    new_row = [row[i] for i in columns_to_keep]
    writer.writerow(new_row)

infile.close()
outfile.close()

file = open('output.csv')
csvreader = csv.reader(file)

header = next(csvreader)
rows = []
filtered_rows = []

count = 0

for row in csvreader:
    rows.append(row)

filtered_rows = []

for row in rows:
    if row[1] != 'Liberal':
        filtered_rows.append(row)
    else:
        if count >  3783:
            filtered_rows.append(row)
        count = count + 1

c_count = 0
l_count = 0

for row in filtered_rows:
    if row[1] == 'Liberal':
        l_count += 1
    else:
        c_count += 1



### Split Dataset Into Multiple Datasets ###

full_dataset = []

for i in filtered_rows:
    if i[1] == 'Liberal':
        full_dataset.append([i[0], 0])
    
    elif i[1] == 'Conservative':
        full_dataset.append([i[0], 1])

random.shuffle(full_dataset)

def split_list(list_1):
    half = len(list_1)//2
    return list_1[:half], list_1[half:]

train_set, test_set = split_list(full_dataset)

train_set_x_unvectorized = []

for i in train_set:
    train_set_x_unvectorized.append(i[0])

train_set_y = []
for i in train_set:
    train_set_y.append(i[1])

test_set_x_unvectorized = []

for i in test_set:
    test_set_x_unvectorized.append(i[0])

test_set_y = []
for i in test_set:
    test_set_y.append(i[1])

train_set_x = []

for i in train_set_x_unvectorized:
    ascii_values = [ord(c) for c in i]
    vectorized_tensor = torch.tensor(ascii_values)

    train_set_x.append(vectorized_tensor)

test_set_x = []

for i in test_set_x_unvectorized:
    ascii_values = [ord(c) for c in i]
    vectorized_tensor = torch.tensor(ascii_values)

    test_set_x.append(vectorized_tensor)

train_set_x = torch.nn.utils.rnn.pad_sequence(train_set_x, batch_first=True)
test_set_x = torch.nn.utils.rnn.pad_sequence(test_set_x, batch_first=True)

print(train_set_x[0].shape)