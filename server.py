import pickle
import socket
import sys
from multiprocessing.pool import ThreadPool


TIMEOUT = 0.2
MAXTHREADS = 16
MAXMSG=256<<10 

def send_message(send_queue,addrports):
    try:
        message=send_queue.popleft()
        message_raw = pickle.dumps(message)

        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
            sock.settimeout(TIMEOUT)
            sock.connect(addrports[message.receiver])
            sock.sendall(message_raw)        
    except:
        pass
      

def send_messages(send_queue,addrports,pool,wait=False):
    tasks = []
    for _ in range(len(send_queue)):
        tasks.append(pool.apply_async(send_message,args=(send_queue,addrports)))
    
    if wait:
        for t in tasks:
            t.wait()



# Server loop, runs in the background
def server_loop(self_proc_id, addrports, recv_queue):
    self_addr_port = addrports[self_proc_id]
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        sock.settimeout(TIMEOUT)

        #Try to bind to specified address and port, may fail if unavailable
        try:
            sock.bind(self_addr_port)
        except:
            print(f'Failed to bind to address {repr(self_addr_port)}')
            sys.stdout.flush()
            sys.exit()

        #Enter the loop
        while True:
            try:                
                # Receive request
                data, _ = sock.recvfrom(MAXMSG)
                message=pickle.loads(data)
                recv_queue.append(message)

            except socket.timeout:
                pass



def send_iter(addrports, send_queue, wait=False):
    try:
        pool = ThreadPool(processes=min(MAXTHREADS,len(addrports)))
        while True:
            send_messages(send_queue, addrports, pool, wait=wait)
            yield None
    finally:
        pool.close()
        pool.join()


