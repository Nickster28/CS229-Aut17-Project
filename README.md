# CS229 - L.A.I.R.

## Overview
Each year, thousands of students enroll in Stanford’s introductory CS106 computer science classes.  Thus, wait times for students at office hours, also referred to as “the LaIR”, can be problematic.  The goal of this project is to take past LaIR request data and use Machine Learning algorithms to predict a student’s total help time: wait time + help time. Providing this estimate to students and teaching staff is helpful for students’ time planning, as well as for the course staff’s ability to estimate the time required to resolve all outstanding questions.  This model may also provide insights into what factors most impact help times, from proximity to homework due date, to time of day, to the type of problem with which the student is having trouble.

## Installation
This project uses Python3.  To create a virtual environment and install all the necessary dependencies, go through the following steps (adapted from Stanford's CS231n [homework setup tutorial](https://cs231n.github.io/assignments2017/assignment1/):

1. *Install Python3*

On a Mac, you can install Python3 by first [installing homebrew](https://brew.sh).  Then, execute `brew install python3` to install Python3.

2. *Install virtualenv*

virtualenv lets you create a "virtual environment" that houses all the necessary requirements for this project, but keeps them isolated from the rest of your system.  To install it, use the Python package manager, `pip`:
```
sudo pip install virtualenv
```

3. *Create a new virtual environment*

Create a virtual environment named `.env` by executing the following command:
```
virtualenv -p python3 .env
```

4. *Working in the virtual environment*

Whenever you would like to work with this project, activate your virtual environment.  You can do this by executing `source .env/bin/activate`.  When you are finished working, you can deactivate your virtual environment by executing `deactivate`.

*Note*: the first time you activate your virtual environment, make sure to install all the necessary requirements for this project.  You can do so by having pip install all packages listed in the requirements file for this project:

```
pip install -r requirements.txt
```





