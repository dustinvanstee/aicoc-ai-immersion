# Dask on Summit


## Running Dask on Summit via Ipython Terminal
You will need 2 terminals and a browser for this lab
___
## terminal 1 setup
1. login to summit
2. activate conda environment<br> 
`module load ibm-wml-ce/1.7.0-1`
`conda activate wmlce17-ornl`

3. launch ipython <br>
`ipython`


## terminal 2 setup
1. forward ssh ports from login node to your laptop.  Here XXXX should be an unused port on the system.  Use 7777 as example<br> Pay attention to making sure the right 
`ssh -N -L XXXX:loginYY.summit.olcf.ornl.gov:XXXX  userid@summit.olcf.ornl.gov`<br>

----
## Terminal 1 ipython

```
import sys
from dask_jobqueue import LSFCluster


### Per node specification
dask_worker_prefix = "jsrun -n1 -a1 -g0 -c2"

cluster = LSFCluster(
    scheduler_options={"dashboard_address": ":5051"},
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

### View LSF bsub job
print(cluster.job_script())

# Create client - https://distributed.dask.org/en/latest/client.html
from dask.distributed import Client
client = Client(cluster)
client
```
## Add some nodes to dask cluster
`cluster.scale(4)`

## Use bjobs to view job status
`watch bjobs`


## Numpy Example
```
import dask.array as da
# 25x10^8 B element array , 400 chunks
x = da.random.random([50000,50000], chunks=[2500,2500])

# lazy execution .
y = x.T ** x - x.mean()

# trigger computation
# Note if you run y.compute() the result is not saved ... 
# each request triggers computation..
print(y.compute())
print(y.compute())

# Now lets pin it to memory ... and re-run
y.persist()
```

## Pandas Example
```
# import dask
import dask.dataframe as dd
ddf = dd.read_csv("./dask-tutorial/ldata2016.csv", blocksize=15e6,dtype=dtype) # , compression="gzip")
#
#ddf = ddf.repartition(npartitions=5)
ddf

# Standard operations example
filtered_df = ddf[ddf["loan_amnt"] > 15000]
answer = filtered_df.compute()
#compare 
len(answer)
len(ddf)


print(ddf.columns)
# ok, lets count NaNs ..
ddf.isna().sum().compute()

# well dask doesnt do well with NaNs,  let just do a few colums ..
ddf_small = ddf[[ 'id', 'loan_amnt', 'funded_amnt','revol_bal','dti']]

# Check NaNs 
ddf_small.isna().sum().compute()

ddf_small.describe().compute()

# correlation
ddf.corr().compute()

# Do one join.. cartesian ?
merge(ddf_small, ddf_small,on='id')# [['loan_amnt', 'funded_amnt']]
```



print(y.compute())



