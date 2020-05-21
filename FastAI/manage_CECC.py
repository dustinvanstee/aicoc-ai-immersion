#!/Users/dustinvanstee/anaconda3/envs/mldl37/bin/python
# manage_h20.py

# Simple script to update python licences ...
import paramiko
import hashlib
from scp import SCPClient
import time,sys
import argparse as ap

# utility print function
def nprint(mystring) :
    print("**{}** : {}".format(sys._getframe(1).f_code.co_name,mystring))


# machine_dict = { 
#   #"vm1" : {"host" :"xxxxxxxxxx", "ip" : "129.40.94.89", "password":"NWuu-NQPRI4zFqA"},
#   "vm2" : {"host" :"p1253-kvm1", "ip" : "129.40.51.65", "password":"077o+w%xkSKT97c"}
# }
  


class remoteConnection:
    """
    remoteConnection- add some comment here
    """
    service_name = 'abstract'
    service_port = 0
    binaryLocation = ""
    install_path = ""

    config_files = {"file1" : "path/to/file1"}

    def __init__(self, server, username, password):
        self.server = server
        self.username = username
        self.password = password
        self.dirName = ""
        self.useLocal = False # Use Local Copy of interpreter.json (instead of trying to download server copy
        self.tarDownloaded = False
        self.connect()

    def connect(self):
        import paramiko
        print("establishing ssh %s @ %s" % (self.username, self.server))
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy())
        self.ssh.connect(self.server, username=self.username, password=self.password)
        self.scp = SCPClient(self.ssh.get_transport())

    def runcmd(self, command, timeout=1000000):
        print("Running : {} ".format(command))
        stdin, stdout, stderr = self.ssh.exec_command(command, timeout=timeout)
        output = stdout.read()
        for myl in output.splitlines():
            print("runcmd : {}".format(myl))


    def setup_private_github(self):

        public_key = 'AAAAB3NzaC1yc2EAAAADAQABAAABAQDGOskWU2Wk9Eejhobdl48gpOjwgX0v6acGH+gK64AeK5gTtFxrPHdjh4T+WZx2YSTtSvKvBwjcCGYB6N0J/+Q3Q85ax5cpS8SBEWSkZ1CcnScCRKR2Zx/n/nBnnKgS/+PstPed7cx0sa/lWGBrC/q51Lgpo0ljbMGpRjnV8wLWnm3r1hX9ioCOI7AFFBQADFxZtwnjSaWMxOrfpXR4Oqb7U/hCIS3fqn7OiTKyqBcEGglCJijGOUIPLl0iEQCGc4lUfQn+KdBbT1TBF1A1U1vs2jNmkUx1NSG3aTm4KRBczw/6TvGJGQY21CpBhKCDGSfg5r2UP16aP6RQA52dO4v/'

        command = 'echo "ssh-rsa ' + public_key + ' dustinvanstee@dustins-mbp.pok.ibm.com" >> ~/.ssh/authorized_keys'

        self.runcmd(command)
        #output = stdout.read()


        print("Transferring github keys")
        if(self.username == 'cecuser') :
            self.runcmd("rm -f /home/cecuser/.ssh/id_rsa")
            self.scp.put( "/Users/dustinvanstee/.ssh/nimbix_id_rsa" , '/home/cecuser/.ssh/id_rsa' )
            self.scp.put( "/Users/dustinvanstee/.ssh/nimbix_id_rsa.pub" , '/home/cecuser/.ssh/id_rsa.pub' )
            self.scp.put( "/Users/dustinvanstee/.ssh/nimbix_config" , '/home/cecuser/.ssh/config' )
            self.scp.put('/Users/dustinvanstee/data/work/osa/2020-05-ai-college-box/labs-demos-ibm-git/FastAI/.condarc.acc', '~/.condarc')
            self.scp.put('/Users/dustinvanstee/data/work/osa/2020-05-ai-college-box/labs-demos-ibm-git/FastAI/setup_fastai.sh', '~/setup_fastai.sh')
            self.scp.put('/Users/dustinvanstee/data/work/osa/2020-05-ai-college-box/labs-demos-ibm-git/FastAI/start_jupyter.sh', '~/start_jupyter.sh')


        stdin, stdout, stderr = self.ssh.exec_command('git config --global user.email "vanstee@us.ibm.com"')
        stdin, stdout, stderr = self.ssh.exec_command('git config --global user.name "Dustin VanStee"')


    def print_login(self):
        print("ssh {0}@{1}".format(self.username,self.server))


class SmartFormatterMixin(ap.HelpFormatter):
    # ref:
    # http://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
    # @IgnorePep8

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('S|'):
            return text[2:].splitlines()
        return ap.HelpFormatter._split_lines(self, text, width)


class CustomFormatter(ap.RawDescriptionHelpFormatter, SmartFormatterMixin):
    '''Convenience formatter_class for argparse help print out.'''


def _parser():
    parser = ap.ArgumentParser(description='Tool to setup fastai in CECC ',
                               formatter_class=CustomFormatter)

    parser.add_argument(
        '--host', action='store', required=True, help='S|--host=host or ip'
             'Default: %(default)s')
    parser.add_argument(
        '--user', action='store', required=True, help='S|--user=username'
             'Default: %(default)s')
    parser.add_argument(
        '--password', action='store', required=True, help='S|--password=password'
             'Default: %(default)s')


    parser.add_argument(
        '--install_code', action='store',  required=False,
        choices=["True","False"], default="False",
        help='S|--force_refresh=[True|False] '
             'Default: %(default)s')

    parser.add_argument(
        '--start_jupyter', action='store', required=False,
        choices=["True","False"], default="False",
        help='S|--force_refresh=[True|False] '
             'Default: %(default)s')

    parser.add_argument(
        '--venv', type=str, default="junk", required=True,
        help='S|Name of virtual environemnt for fastai'
             'Default: %(default)s')

    args = parser.parse_args()

    return args



def main() :
    args = _parser()

    for argk in vars(args) :
        print(argk,vars(args)[argk])

    myConn = remoteConnection(args.host,args.user,args.password)
    
    if(args.install_code=="True") : 
        myConn.setup_private_github()
        myConn.runcmd('conda create -y -n {}'.format(args.venv))
        myConn.runcmd('conda activate {}; bash ./setup_fastai.sh'.format(args.venv))
    
    if(args.start_jupyter =="True"): 
        try :
            print("Starting Jupyter !")
            myConn.scp.put('/Users/dustinvanstee/data/work/osa/2020-05-ai-college-box/labs-demos-ibm-git/FastAI/notebook.json', '~/.jupyter/nbconfig/notebook.json')
            myConn.runcmd('conda activate {}; bash ./start_jupyter.sh'.format(args.venv),timeout=10)
        except  :
            print("Command timeout .. its ok its expected !")

    #myConn.runcmd('cat ~/.ssh/id_rsa.pub')
    myConn.print_login()

if __name__ == "__main__":
    main()
   


 
    #stdin, stdout, stderr = myssh.exec_command("sudo cat /etc/dai/.htpasswd") 
    #print(stdout.read())

#    setup_env(skey, "powerai-test")



# VM 7 created Sept 1 2019
#create_id("vm7", "user668", 'XFaM96rS%0BK69_')
# VM 1-4 created Sept 23 2019
#create_id("vm1", "user668", 'XFaM96rS%0BK69_')
#create_id("vm2", "user668", 'XFaM96rS%0BK69_')
#create_id("vm3", "user668", 'XFaM96rS%0BK69_')
#create_id("vm4", "user668", 'XFaM96rS%0BK69_')
#create_id("vm5", "user668", 'XFaM96rS%0BK69_')
#create_id("vm6", "user668", 'XFaM96rS%0BK69_')
#create_id("vm8", "user668", 'XFaM96rS%0BK69_')



#
#   stdin, stdout, stderr = myssh.exec_command('sudo chown root:root /etc/dai/config.toml') 
#    print("chown config.toml")
#    #had to do this 2x for some reason and dont feel like debuggin
#    stdin, stdout, stderr = myssh.exec_command('sudo chown root:root /etc/dai/config.toml') 
#    print(stdout, stderr)
#    #add vanstee to all machines ..
#    stdin, stdout, stderr = myssh.exec_command('sudo htpasswd -bBc "/etc/dai/.htpasswd" vanstee passw0rd') 
#
#    #auto generated
#    for i in range(20) :
#        h20userid    = "user_{:02}".format(i)
#        h20password  = hashlib.md5(bytes(h20userid+server, "utf-8")).hexdigest()[0:4]
#        expiry       = "01-01-2020"
#        time.sleep(1)
#        stdin, stdout, stderr = myssh.exec_command('sudo htpasswd -bB "/etc/dai/.htpasswd" ' + h20userid + ' ' + h20password)
#        #stdin, stdout, stderr = myssh.exec_command("sudo cat /etc/dai/.htpasswd") 
#        #print(stdout.read())
#        print("Created userid {} / {} on {}:12345 expires {}".format(h20userid, h20password, server, expiry))
#       
#    #custom
#    for user_data in mgmt_info_list[vmid] :
#    
#        h20userid    = user_data[0]
#        h20password  = user_data[1]
#        expiry       = user_data[3]
#
#        stdin, stdout, stderr = myssh.exec_command('sudo htpasswd -bB "/etc/dai/.htpasswd" ' + h20userid + ' ' + h20password)
#        #stdin, stdout, stderr = myssh.exec_command("sudo cat /etc/dai/.htpasswd") 
#        #print(stdout.read())
#        print("Created userid {} / {} on {}:12345 expires {}".format(h20userid, h20password, server, expiry))
#
#    time.sleep(1)
#    print("Stopping DAI")
#    stdin, stdout, stderr = myssh.exec_command('sudo systemctl stop dai')
#    print("Starting DAI")
#    stdin, stdout, stderr = myssh.exec_command('sudo systemctl start dai')
#
# #