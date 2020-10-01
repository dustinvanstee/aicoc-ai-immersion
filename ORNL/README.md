
# ML/DL Workshop - ORNL Agenda
## Oct 7/8 2020
___

**Day1**
Session | Date / Time
------------ | -------------
Welcome & Introduction	|11:00 AM - 11:15 AM
A Brief Overview of ML/DL	|11:15 AM - 11:45 AM
Intersection of AI and HPC	|11:45 AM - 12:15 PM
Summit System Overview	|12:15 PM - 12:30 PM
Python Environments on Summit	|12:30 PM - 1:00 PM
Break	|1:00 PM - 1:15 PM
Introduction to OPEN-CE	|1:15 PM - 1:45 PM 
Lab 0: Getting Started on Summit	|1:45 PM - 2:00 PM
Lab 1: Python AI Essentials for Beginners	|2:00 PM - 3:30 PM
Break	|3:30 PM - 3:45 PM
Lab 2: GPU accelerated dataframes - Rapids Lab 	                         |3:45 PM - 5:15 PM
*****************************************************************************************************|*****

**Day2**
Session | Date / Time
------------ | -------------
Welcome Back	|11:00 AM - 11:15 AM
Overview of Neural Networks 	|11:15 AM - 11:45 AM
Lab 3: Intro to PyTorch (w/lecture)	|11:45 AM - 1:00 PM
Break	|1:00 PM - 1:15 PM
Integrating Trained Models into C++/FORTRAN Applications	|1:15 PM - 1:45 PM
Lab 4: E2E Deep Learning (Training) Using Deep Learning as a universal function approximator	|1:45 PM - 3:00 PM
Break	|3:00 PM - 3:15 PM
Lab 5: E2E Deep Learning (Inference) Function Representation Integrated into Example Application |3:15 PM - 5:00 PM
Closing Remarks	|5:00 PM - 5:15 PM
*****************************************************************************************************|*****

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


