''' 
Split learning (Client 2 -> Server -> Client 2)
'''
import gc
from glob import escape
import os
from pyexpat import model
import socket
import struct
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import sys
import copy
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from time import process_time

import MyNet2
import socket_fun as sf

# variável global
MSG = b'ok!'  
MODE = 0    # 0->treinar, 1->testar

BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("device: ", device)

# define transformações
root = './models/cifar10_data'
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),])

# baixar conjunto de dados
trainset = torchvision.datasets.CIFAR10(root=root, download=True, train=True, transform=transform)

# se você quiser dividir o conjunto de dados, remova os comentários abaixo.
# indices = np.arange(len(trainset))
# train_dataset = torch.utils.data.Subset(
#     trainset, indices[0:20000]
# )
# trainset = train_dataset
testset = torchvision.datasets.CIFAR10(root=root, download=True, train=False, transform=transform)

print("trainset_len: ", len(trainset))
print("testset_len: ", len(testset))
image, label = trainset[0]
print (image.size())

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

mymodel1 = MyNet2.MyNet_in().to(device)
mymodel2 = MyNet2.MyNet_out(NUM_CLASSES=10).to(device)

# -------------------- conexão ----------------------
# estabelecer socket
host = 'localhost'
port = 19089
ADDR = (host, port)

# CONECTAR
s = socket.socket()
s.connect(ADDR)
w = 'start communication'

# ENVIAR
s.sendall(w.encode())
print("solicitação enviada ao servidor")
dammy = s.recv(4)

# ------------------ iniciar treinamento -----------------
epochs = 5
lr = 0.005

# definir função de erro
criterion = nn.CrossEntropyLoss()

# definir otimizador
optimizer1 = optim.SGD(mymodel1.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(mymodel2.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

def train():
    # função de forward propagation
    def forward_prop(MODEL, data):

        output = None
        if MODEL == 1:
            optimizer1.zero_grad()
            output_1 = mymodel1(data)
            output = output_1
        elif MODEL == 2:
            optimizer2.zero_grad()
            output_2 = mymodel2(data)
            output = output_2
        else:
            print("!!!!! MODELO não encontrado !!!!!")

        return output

    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    p_time_list = []

    for e in range(epochs):
        print("--------------- Época: ", e, " --------------")
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        p_start = process_time()
        # ================= modo treino ================
        mymodel1.train()
        mymodel2.train()
        ### enviar MODO
        MODE = 0    # modo treino -> 0:treinar, 1:testar, 2:terminou treino, 3:terminou teste
        MODEL = 1   # modelo de treino
        for data, labels in tqdm(trainloader):
            # enviar número do MODO
            sf.send_size_n_msg(MODE, s)
        
            data = data.to(device)
            labels = labels.to(device)

            output_1 = forward_prop(MODEL, data)

            # ENVIAR ----------- dados de características 1 ---------------
            sf.send_size_n_msg(output_1, s)
            
            
            

            ### aguardar o SERVIDOR calcular... ###
            
            
            
            

            # RECEBER ------------ dados de características 2 -------------
            recv_data2 = sf.recv_size_n_msg(s)
            
            MODEL = 2   # receber dados de características 2 -> MODELO=2

            # iniciar forward propagation 3
            OUTPUT = forward_prop(MODEL, recv_data2)

            loss = criterion(OUTPUT, labels)

            loss.backward()     # partes out-layer

            optimizer2.step()

            # ENVIAR ------------- grad 2 -----------
            sf.send_size_n_msg(recv_data2.grad, s)

            # RECEBER ----------- grad 1 -----------
            recv_grad = sf.recv_size_n_msg(s)

            MODEL = 1

            train_loss = train_loss + loss.item()
            train_acc += (OUTPUT.max(1)[1] == labels).sum().item()

            output_1.backward(recv_grad)    # partes in-layer

            optimizer1.step()
        
        avg_train_loss = train_loss / len(trainloader.dataset)
        avg_train_acc = train_acc / len(trainloader.dataset)
        
        print("modo treino finalizado!!!!")
        
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)

        # =============== modo teste ================
        mymodel1.eval()
        mymodel2.eval()

        with torch.no_grad():
            print("iniciar modo teste!")
            MODE = 1    # mudar modo para teste
            for data, labels in tqdm(testloader):
                # enviar número do MODO
                sf.send_size_n_msg(MODE, s)

                data = data.to(device)
                labels = labels.to(device)
                output = mymodel1(data)

                # ENVIAR --------- dados de características 1 -----------
                sf.send_size_n_msg(output, s)

                ### aguardar o servidor...

                # RECEBER ----------- dados de características 2 ------------
                recv_data2 = sf.recv_size_n_msg(s)

                OUTPUT = mymodel2(recv_data2)
                loss = criterion(OUTPUT, labels)

                val_loss += loss.item()
                val_acc += (OUTPUT.max(1)[1] == labels).sum().item()

        p_finish = process_time()
        p_time = p_finish-p_start
        if e == epochs-1:
            MODE = 3
            # sf.send_size_n_msg(MODE, s)     # por cliente ver.
        else:                                 # por época ver.
            MODE = 2    # terminou teste -> iniciar treino do próximo cliente
        sf.send_size_n_msg(MODE, s)           # por época ver.
        print("Tempo de processamento: ", p_time)
        p_time_list.append(p_time)

        avg_val_loss = val_loss / len(testloader.dataset)
        avg_val_acc = val_acc / len(testloader.dataset)
        
        print ('Epoch [{}/{}], Loss: {loss:.5f}, Acc: {acc:.5f},  val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}' 
                    .format(e+1, epochs, loss=avg_train_loss, acc=avg_train_acc, val_loss=avg_val_loss, val_acc=avg_val_acc))
        
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

        if e == epochs-1: 
            s.close()
            print("Conexão de socket finalizada (CLIENTE 3)")

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list

import csv

def write_to_csv(train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list):
    file = './path/client2.csv'
    f = open(file, 'a', newline='')
    csv_writer = csv.writer(f)

    result = []
    result.append('client 2')
    result.append(train_loss_list)
    result.append(train_acc_list)
    result.append(val_loss_list)
    result.append(val_acc_list)
    result.append(p_time_list)

    csv_writer.writerow(result)

    f.close()

if __name__ == '__main__':
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list = train()
    write_to_csv(train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list)
