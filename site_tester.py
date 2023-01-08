import csv
import random
from transformers import pipeline
from readability import Document

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import requests


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
        full_dataset.append([i[0], 0.0])
    
    elif i[1] == 'Conservative':
        full_dataset.append([i[0], 1.0])

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

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

input_size = 300
hidden_size = 100
output_size = 1

model = BinaryClassifier(input_size, hidden_size, output_size)

# define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_set_x = torch.tensor(train_set_x)
train_set_y = torch.tensor(train_set_y)
train_set_y = train_set_y.view(-1, 1)
test_set_x = torch.tensor(test_set_x)
test_set_y = torch.tensor(test_set_y)
test_set_y = test_set_y.view(-1, 1)


def train_model():
    # training loop
    for epoch in range(75):
        for i in range(len(train_set_x)):
            # reshape inputs to (batch_size, input_size)
            inputs = train_set_x.view(-1, input_size)
            inputs = inputs.float()
            
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, train_set_y)
            
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss)

    torch.save(model.state_dict(), 'model.pth')


def get_political_stance(text):
    model_1 = BinaryClassifier(input_size, hidden_size, output_size)
    model_1.load_state_dict(torch.load('model.pth'))

    ascii_values = [ord(c) for c in text]

    test_set_x_list = test_set_x.tolist()
    test_set_x_list.append(ascii_values)

    test_set_x_tensors = [torch.tensor(lst) for lst in test_set_x_list]
    tensor_test = rnn_utils.pad_sequence(test_set_x_tensors)
    tensor = torch.tensor(tensor_test)
    print(tensor.shape)
    tensor = tensor.float()
    tensor = tensor.view(4536, 300)
    output = model_1(tensor[-1])
    prediction = (output > 0.5).long()
    print(prediction.item())
    return prediction.item()


def test_site(link):

    page = requests.get(link)
    doc = Document(page.content)
    text = doc.summary()

    summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")
    summary_text = summarizer(text, max_length=50, min_length=5, do_sample=False)[0]['summary_text']
    print(summary_text)


    if get_political_stance(summary_text) == 0:
        print('This site leans toward a Liberal viewpoint')
    
    elif get_political_stance(summary_text) == 1:
        print('This site leans to a Conservative viewpoint')


def get_link_site():

    link = input('What is your site link? \n')
    test_site(link)