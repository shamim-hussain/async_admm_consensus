import json
import sys

if __name__ == '__main__':
    num_worker = 2
    start_port = 20000
    output_file = 'knownhosts.json'
    
    if len(sys.argv)>1:
        num_worker = int(sys.argv[1])
    
    if len(sys.argv)>2:
        start_port = int(sys.argv[2])
    
    if len(sys.argv)>3:
        output_file = sys.argv[3]
    
    hosts = {}
    
    for i in range(num_worker+1):
        h = {i:{'udp_start_port':start_port+i, 'ip_address': "127.0.0.1"}}
        hosts.update(h)
    
    data = {'hosts':hosts}
    
    with open(output_file, 'w') as file:
        json.dump(data, file, indent='\t')
    