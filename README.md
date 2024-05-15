# IncomeSim

IncomeSim is a time-series simulator based on the [Adult dataset](http://archive.ics.uci.edu/dataset/2/adult).

## Prerequisites

* IncomeSim is written in Python 3 and based on the Scikit-learn package and the Adult dataset. 
* Start by installing python modules ```pandas, numpy, scikit-learn, jupyter, matplotlib, yaml, xgboost```, for example in a virtual environment. Below, we list the versions used during development and testing. 
  ```
  pip install scikit-learn==1.4.1.post1 pandas==2.0.1 PyYAML==6.0 xgboost==2.0.0
  ```

## Using the simulator

* The IncomeSCM simulator is fit to the [Adult dataset](http://archive.ics.uci.edu/dataset/2/adult) data set.
* To fit the simulator, run the python script ```fit.py```
```
python fit.py [-c CONFIG_FILE]
```
* The default config file is configs/config_v1.yml
* To sample from the simulator, use the script ```sample.py```
```
python fit.py [-c CONFIG_FILE]
```
* This also uses the same default config file, which specifies which fitted model to use, how many samples are used, and from which (counterfactual) policy to sample. By default, 50 000 samples are generated from the "default" (observational) "full" and "no" policies. 

## Papers using the data set 

## Lectures using the data set 

### DAT465 Lecture [2023]

If you want to follow along in the notebook during the demo lecture
1. Clone this repository
2. Install prerequisites

For example using a virtual environment: 
```bash
virtualenv dat465
source dat465/bin/activate
pip install pandas numpy scikit-learn jupyter matplotlib
```

The slides for the lecture can be found on Canvas.

## Coding in the demo

* Open [dat465_lecture_demo.ipynb](demos/dat465_lecture_demo.ipynb) in Jupyter in a Python environment with the prerequisites above
```bash
jupyter notebook   
```

### ProbAI 23 lecture [2023]

If you want to follow along in the notebook during the ProbAI lecture, you have two options: 
1. Clone this repository and open [probai_lecture_github.ipynb](demos/probai_lecture_github.ipynb) in Jupyter/Jupyter lab
2. Work in Colab from this [notebook](https://colab.research.google.com/drive/1jlEsSYcCDiqhamshxhkdQ703KKWaJHL9?usp=sharing)

The slides for the lecture can be found [here](demos/ProbAI_Causal_machine_learning.pdf).

**Installing prerequisites**

* IncomeSim is written in Python 3 and based on the Scikit-learn package and the Adult dataset. 
* Start by installing python modules ```pandas, numpy, scikit-learn, jupyter, requests, matplotlib```

**Preparing the data files** 

You don't need to do this if you use the ProbAI notebook, the notebook does this automatically!

* Download the [Adult dataset](http://archive.ics.uci.edu/dataset/2/adult)
* Create a folder ``` data/income ``` in the IncomeSim root folder
* Place the files ``` adult.data ```, ``` adult.names ``` and ``` adult.test ``` in ``` data/income ```

**Generating data**

* Run ``` python generate.py -n <number of samples> -T <length of horizon> ``` to fit the simulator and generate data
