from paramiko import SSHClient
from scp import SCPClient


ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect(hostname='ip', 
            port = 'port',
            username='seleznev.m',
            password='password',
            pkey='load_key_if_relevant')


# SCPCLient takes a paramiko transport as its only argument
scp = SCPClient(ssh.get_transport())

#scp.put('file_path_on_local_machine', 'file_path_on_remote_machine')
#scp.get('file_path_on_remote_machine', 'file_path_on_local_machine')

scp.close()