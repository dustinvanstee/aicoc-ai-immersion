# Dask on Summit
00_dask_on_summit.ipynb

## Running Jupyter Notebook
This step is not required for the lab but mainly informational.  Note you should not be running anything on login nodes that require heavy compute or GPU as this is a highly shared resource.  This is mainly for viewing or light debug/testing.
___
To view a jupyter notebook from the login node you need two terminals
In terminal 1 
1. login to summit
2. activate conda environment (as shown above)
3. launch jupyter notebook or jupyter lab <br>
jupyter notebook --no-browser --ip=\`hostname -f\` --port=[PORT]


In terminal 2 
1. forward ssh ports from login node to your laptop<br>
ssh -N -L XXXX:loginYY.summit.olcf.ornl.gov:XXXX  userid@summit.olcf.ornl.gov


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



