

import numpy as np
import torch
from admm import Message, Worker, Master
import socket
import sys
import threading
import json
from server import server_loop, send_iter
#from types import SimpleNamespace



KNOWNHOSTFILE='knownhosts.json'


class AvgWorker(Worker):
    def __init__(self, X_i, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X_i = X_i
        
    def local_optim(self):
        self.x_i = (self.X_i.mean(0)+self.beta*self.z-self.l_i)/(1+self.beta)



def objective(X,z):
    return 0.5*((X-z)**2).mean(0).sum()







# Read knownhosts.json
def read_json(site_id, filename):
    try:
        with open(filename,'r') as fp:
            deserialized=json.load(fp)
    except:
        print('Could not open json file!')
        sys.stdout.flush()
        sys.exit()
    
    addrports=[]
    self_proc_id = 0
    hosts_info=deserialized['hosts']
    hosts = {}

    for i, host in enumerate(sorted(hosts_info.keys())):
        hosts[i]=host
        if host==site_id:
            self_proc_id = i
        info = hosts_info[host]
        addrports.append((info['ip_address'], info['udp_start_port']))
    
    return self_proc_id, addrports, hosts





def run_master(self_proc_id, addrports):
    num_worker = len(addrports)-1
    beta = 0.1
    S = num_worker
    tau = 1
    device = 'cpu:0'
    steps = 100
    
    with np.load('mnist.npz') as dat:
        X = dat['X_train']
        
    X = torch.from_numpy(X).float().to(device)/255
    x_dim = tuple(X.shape[1:])
    
    master = Master(num_worker, x_dim, beta, S, tau, device)
    
    
    # Start the server loop as a daemon thread
    sever_thread=threading.Thread(target=server_loop,
                                  args=(self_proc_id, addrports,
                                        master.send_queue), 
                                  daemon=True)
    sever_thread.start()
    
    
    obj_vals = []
    
    for _ in send_iter(addrports, master.send_queue, wait=False):           
        master.receive()
        if master.update():
            obj_vals.append(objective(X,master.z))
            if not master.k % 5:
                print(f'Step {master.k} : Objective = {obj_vals[-1]:.5f}')
        
        if master.k == steps:
            break



def run_worker(self_proc_id, addrports):
    num_worker = len(addrports)-1
    beta = 0.1
    device = 'cpu:0'
    steps = 100
    w_i = self_proc_id
    
    with np.load('mnist.npz') as dat:
        X = dat['X_train']
        
    X = torch.from_numpy(X).float().to(device)/255
    x_dim = tuple(X.shape[1:])
    
    
    N = X.shape[0]
    N_i = (N+num_worker-1)//num_worker
    
    begin = min(N_i*w_i,N)
    end = min(N_i*(w_i+1),N)
    X_i = X[begin:end]
    worker = AvgWorker(X_i, w_i, num_worker, x_dim, beta, device)
    
    
    # Start the server loop as a daemon thread
    sever_thread=threading.Thread(target=server_loop,
                                  args=(self_proc_id, addrports,
                                        worker.send_queue), 
                                  daemon=True)
    sever_thread.start()
    
    
    for _ in send_iter(addrports, worker.send_queue, wait=False):           
        worker.receive()
        if worker.update():
            pass
        
        if worker.k == steps:
            break


# Main function
def main():
    if len(sys.argv) < 2:
        print('Warning: Site ID not provided, using system hostname.')
        site_id=socket.gethostname()
    else:
        site_id=sys.argv[1]
    
    # Read json file
    self_proc_id,addrports,_ = read_json(site_id,KNOWNHOSTFILE)
    
    if self_proc_id == len(addrports)-1:
        run_master(self_proc_id,addrports)
    else:
        run_worker(self_proc_id,addrports)




if __name__ == '__main__':
    main()
        