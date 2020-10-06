
# ML/DL Workshop - ORNL Agenda
## Oct 7/8 2020
___

# Lab0 :Conda Environment Setup
___
To setup an environment that can run the labs shown, you will need to use IBM's WML-CE (now called OPEN-CE) module with some additional packages.  To create your own conda virtual environment with all the dependencies, you can use the provided requirements.txt file in this repo with the following commands.

1.  Clone this repo and cd to ORNL directory

2.  Load WML-CE to get access to conda<br>
`module load ibm-wml-ce/1.7.0-1`

3. Create a conda virtual environment cloned off most recent WML-CE with extra pkgs (accept terms by pressing 1 + enter) <br>
`conda env create -f wmlce17-ornl.yml`

If all goes well after this process, you can start you conda environment by running<br> `conda activate wmlce17-ornl`

We will be using this environment to submit batch jobs to the cluster.

# Lab1 : Submit AI Essentials Notebook
___
1. cd to this directory 
`cd ORNL`

2. Batch submission to LSF
`bsub 01_submit_aiessentials.lsf`

# Lab2 : Submit Rapids example
___
1. cd to this directory 
`cd ORNL`

2. Batch submission to LSF
`bsub 02_submit_rapids.lsf`

# Lab3 : Submit Pytorch Image Classifier
___
1. cd to this directory 
`cd ORNL`

2. Batch submission to LSF
`bsub 03_submit_pytorch.lsf`

# Lab4 : Submit Universal Function Approximator Training
___
1. cd to this directory 
`cd ORNL`

2. Batch submission to LSF
`bsub 04_submit_ufa_train.lsf`

# Lab5 : Submit Universal Function Approximator Inference
___
1. cd to this directory 
`cd ORNL`

2. Batch submission to LSF
`bsub 05_submit_ufa_inf.lsf`

# Dask demo 

[link to readme](https://github.com/dustinvanstee/aicoc-ai-immersion/ORNL/blob/branch/other_file.md)


