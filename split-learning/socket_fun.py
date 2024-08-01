import pickle

MSG = b'ok!'

def recv_size_n_msg(s):
    exp_size = int(s.recv(16))
    s.sendall(MSG)
    recv_size = 0
    recv_data = b''
    while recv_size < exp_size:
        packet = s.recv(524288)
        # packet = s.recv(4096)
        recv_size = recv_size + len(packet)
        recv_data = recv_data + packet

    s.sendall(MSG)
    recv_data = pickle.loads(recv_data)

    return recv_data

def send_size_n_msg(msg, s):
    bytes = pickle.dumps(msg)
    msg_size = len(bytes)
    msg_size_bytes = str(format(msg_size, '16d')).encode()
    s.sendall(msg_size_bytes)
    dammy = s.recv(4)
    s.sendall(bytes)
    dammy = s.recv(4)
