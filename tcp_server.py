import pickle
import socket
import sys
from multiprocessing.pool import ThreadPool
import time


TIMEOUT = 0.2
MAXTHREADS = 16
MAXMSG=256<<10 


class Server:
    def __init__(self, addrport, send_queue, recv_queue, num_workers):
        self.addrport = addrport
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self.num_workers = num_workers
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind(self.addrport)
            self.sock.listen()
            print(f'Started listening on address {repr(self.addrport)}')
        except:
            raise Exception(f'Failed to bind to address {repr(self.addrport)}')
        
    def accept(self):
        self.conns = {}
        for _ in range(self.num_workers):
            conn, _ = self.sock.accept()
            i = pickle.loads(conn.recv(MAXMSG))
            conn.setblocking(False)
            self.conns.update({i:conn})
            print(f'Connection {i} connected.')
    
    
    
    def recv_message(self, data):
        message=pickle.loads(data)
        self.recv_queue.append(message)
    
    def recv_loop(self):
        try:
            pool = ThreadPool(processes=min(MAXTHREADS,self.num_workers))
            while True:
                for i, conn in self.conns.items():
                    try:
                        data = conn.recv(MAXMSG)
                        pool.apply_async(self.recv_message,
                                         args=(data,))
                    except BlockingIOError:
                        pass
                    except ConnectionResetError:
                        print(f'Connection {i} closed.')
                        return
        finally:
            pool.close()
            pool.join()
    
    
    def send_message(self):
        try:
            message=self.send_queue.popleft()
        except:
            return None
        
        try:
            self.conns[message.receiver].sendall(pickle.dumps(message))
        except ConnectionAbortedError:
            print(f'Warning: Trying to send message after connection {message.receiver} was closed!')
    
    
    def send_iter(self):
        try:
            pool = ThreadPool(processes=min(MAXTHREADS,self.num_workers))
            while True:
                for _ in range(len(self.send_queue)):
                    pool.apply_async(self.send_message)
                    
                yield None
        finally:
            pool.close()
            pool.join()
          
    
    def __del__(self):
        print(f'Stop listening on address {self.addrport}')
        for conn in self.conns.values():
            conn.close()
            
        self.sock.close()













class Client:
    def __init__(self, addrport, send_queue, recv_queue, num_workers, worker_id, max_retry_connect=3):
        self.addrport = addrport
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self.num_workers = num_workers
        self.worker_id = worker_id
        
        for tt in range(max_retry_connect+1):
            if tt==max_retry_connect:
                raise Exception(f'Worker:{self.worker_id} failed to connect to address {repr(self.addrport)}')
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(self.addrport)
                self.sock.sendall(pickle.dumps(self.worker_id))
                print(f'Worker:{self.worker_id} connected to address {repr(self.addrport)}')
            except:
                print(f'Worker:{self.worker_id} failed to connect to address {repr(self.addrport)}. Retrying in 1 second...')
                time.sleep(1)
            else:
                break
        
    
    def recv_message(self, data):
        message=pickle.loads(data)
        self.recv_queue.append(message)
    
    
    def recv_loop(self):
        try:
            pool = ThreadPool(processes=min(MAXTHREADS,self.num_workers))
            while True:
                data = self.sock.recv(MAXMSG)
                pool.apply_async(self.recv_message,
                                 args=(data,))

        finally:
            pool.close()
            pool.join()
    
    
    def send_message(self):
        try:
            message=self.send_queue.popleft()
        except:
            return None
        
        self.sock.sendall(pickle.dumps(message))
    
    
    def send_iter(self):
        try:
            pool = ThreadPool(processes=min(MAXTHREADS,self.num_workers))
            while True:
                for _ in range(len(self.send_queue)):
                    pool.apply_async(self.send_message)
                    
                yield None
        finally:
            pool.close()
            pool.join()
          
    
    def __del__(self):
        print(f'Disconnecting from address {repr(self.addrport)}')
        self.sock.close()


