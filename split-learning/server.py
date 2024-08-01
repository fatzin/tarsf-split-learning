''' 
Split learning (Client A -> Server -> Client A)
Server program
'''

from email.generator import BytesGenerator
import os
from pyexpat import model
import socket
import struct
import pickle

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
MSG = b'ok!'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("dispositivo: ", device)

mymodel = MyNet2.MyNet_hidden().to(device)
print("modelo: ", mymodel)

# -------------------- connection ----------------------
# connection establish
user_info = []
host = '0.0.0.0'
port = 19089
ADDR = (host, port)
s = socket.socket()
s.bind(ADDR)
USER = 3
s.listen(USER)
print("Esperando clientes...")

# CONECTAR
for num_user in range(USER):
    conn, addr = s.accept()
    user_info.append({"nome": "Client "+str(num_user+1), "conn": conn, "addr": addr})
    print("Conectado com Cliente "+str(num_user+1), addr)

# RECEBER
for user in user_info:
    recvreq = user["conn"].recv(1024)
    print("receber mensagem do cliente <{}>".format(user["addr"]))
    user["conn"].sendall(MSG)  

# ------------------- start training --------------------
def train(user):

    p_start = process_time()

    i = 1
    ite_counter = -1
    user_counter = 0
    PATH = []
    PATH.append('./savemodels/client1.pth')
    PATH.append('./savemodels/client2.pth')
    PATH.append('./savemodels/client3.pth')
    lr = 0.005
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    LOADFLAG = 0

    while True:
        ### MODO receber
        recv_mode = sf.recv_size_n_msg(user["conn"])

        # ============= modo treino ============
        if recv_mode == 0:
            mymodel.train()
            if LOADFLAG == 1:
                mymodel.load_state_dict(torch.load(PATH[user_counter-1]))
                LOADFLAG = 0

            ite_counter+=1
            print("(USER {}) TRAIN Loading... {}".format(i, ite_counter))

            # RECEBER ---------- data 1 ---------
            recv_data1 = sf.recv_size_n_msg(user["conn"])
            optimizer.zero_grad()

            # Forward prop. 2
            output_2 = mymodel(recv_data1)

            # ENVIAR ------------ data 2 ----------
            sf.send_size_n_msg(output_2, user["conn"])


            # RECEBER ------------ grad 2 ------------
            recv_grad = sf.recv_size_n_msg(user["conn"])

            # Back prop.
            output_2.backward(recv_grad)

            # update param.
            optimizer.step()

            # ENVIAR ------------- grad 1 -----------
            sf.send_size_n_msg(recv_data1.grad, user["conn"])

        # ============= test mode =============
        elif recv_mode == 1:
            ite_counter = -1
            mymodel.eval()
            print("(USER {}) TEST Carregando...".format(i))

            # RECEBER ---------- data 1 -----------
            recv_data = sf.recv_size_n_msg(user["conn"])

            output_2 = mymodel(recv_data)

            # ENVIAR ---------- data 2 ------------
            sf.send_size_n_msg(output_2, user["conn"])

        # =============== Ir para o proximo cliente =============
        elif recv_mode == 2: 
            ite_counter = -1
            torch.save(mymodel.state_dict(), PATH[user_counter-1])
            LOADFLAG = 1
            print(user["name"], " terminou o treinamento!!!")
            i = i%USER
            print("Agora usuário ", i+1)
            user = user_info[i]
            i += 1
        
        # ============== cliente completou, mover para o próximo cliente ==========
        elif recv_mode == 3:
            user_counter += 1
            i = i%USER
            torch.save(mymodel.state_dict(), PATH[user_counter-1])
            LOADFLAG = 1
            print(user["name"], "terminou!!!!")
            user["conn"].close()
            if user_counter == USER: break
            user = user_info[i]
            i += 1

        else:   print("!!!!! erro de MODE !!!!!")

    print("=============Treinamento finalizado!!!!!!===========")
    print("Conexão de socket finalizada (SERVIDOR)")

    p_finish = process_time()

    print("Tempo de processamento: ",p_finish-p_start)

if __name__ == '__main__':
    train(user_info[0])