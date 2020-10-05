# Dask on Summit


## Running Dask on Summit via Ipython Terminal
You will need 2 terminals and a browser for this lab
___
In terminal 1 
1. login to summit
2. activate conda environment 
`conda activate wmlce17-ornl`

3. launch ipython <br>
ipython


In terminal 2
1. forward ssh ports from login node to your laptop.  Here XXXX should be an unused port on the system.  Use 7777 as example<br> Pay attention to making sure the right 
ssh -N -L XXXX:loginYY.summit.olcf.ornl.gov:XXXX  userid@summit.olcf.ornl.gov

[WIP]

ipython

import sys
sys.path.append("/gpfs/alpine/ven201/world-shared/vanstee/dv/dask-jobqueue")

from dask_jobqueue import LSFCluster
dask_worker_prefix = "jsrun -n1 -a1 -g0 -c1"
cluster = LSFCluster(
    scheduler_options={"dashboard_address": ":3762"},
    cores=8,
    processes=1,     
    memory="4 GB",
    project="VEN201",
    walltime="00:30",
    job_extra=["-nnodes 1"],          # <--- new!
    header_skip=["-R", "-n ", "-M"],  # <--- new!
    interface='ib0',
    use_stdin=False,
    python= f"{dask_worker_prefix} {sys.executable}"
)

print(cluster.job_script())

from dask.distributed import Client
client = Client(cluster)



import dask.array as da
# 2.5 B element array , 500 chunks
x = da.random.random([50000,50000], chunks=[2500,2500])



