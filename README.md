# IncomeSCM

IncomeSCM is a time-series simulator based on the [Adult dataset](http://archive.ics.uci.edu/dataset/2/adult) intended for evaluation of causal effect estimators.

It has been used to construct a cross-sectional benchmark data set for conditional average treatment effect (CATE) estimation, IncomeSCM-1.0.CATE. The files for this benchmark are contained in [IncomeSCM-1.0.CATE.zip](samples/IncomeSCM-1.0.CATE.zip).

To replicate the results in [ADD PAPER WHEN AVAILABLE], download this repository, ensuring that required prerequisites are installed and run: 
```console
pip install scikit-learn==1.4.1.post1 pandas==2.0.1 PyYAML==6.0 xgboost==2.0.0 matplotlib==3.7.1
git clone git@github.com:Healthy-AI/IncomeSCM.git

python fit.py
python sample.py
python estimate.py
python present_results.py
```

## Using the CATE estimation data set (IncomeSCM-1.0.CATE)

The IncomeSCM-1.0.CATE data set is sampled from the IncomeSCM-1.0 simulator, fit with the ```config_v1.yml``` configuration file.

### Data set description

The data set represents 13 variables extracted from the 1994 US Census bureau database, as well as a hand-designed "studies" variable. 

**Covariates, $X$**

| Column  | Description | Type |
| ------------- | ------------- | ------------- |
| native-country  | Native country  | Categorical |
| sex  | Sex (as reported in census) | Categorical |
| race  | Race (as reported in census)  | Categorical |
| age  | Age  | Numeric |
| education  | Education type (e.g., Bachelors)  | Categorical |
| education-num  | Education (numeric representation)  | Numeric |
| workclass  | Workclass (e.g., private, self-employed)  | Categorical |
| occupation  | Occupation (e.g., Tech-support)  | Categorical |
| marital-status  | Marital status (e.g., married)  | Categorical |
| relationship  | Relationship type (e.g., wife)  | Categorical |
| capital-net  | Net capital gains  | Numeric |
| hours-per-week  | Number of work hours per week | Numeric |
| income_prev  | Income the previous year (USD)  | Numeric |
| studies_prev  | Studies the previous year  | Categorical |

**Intervention, $A$**
| Column  | Description | Type |
| ------------- | ------------- | ------------- |
| studies  | Studies the current year (e.g., Full-time studies)  | Categorical |

**Outcome, $Y$**
| Column  | Description | Type |
| ------------- | ------------- | ------------- |
| income  | Income 5 years after the intervention (USD)   | Numeric |

### Task description

The goal is to use observational data to estimate the causal effect on ```income``` ($Y$) after intervening on ```studies``` with "Full-time studies" ($A \leftarrow 1$), relative to "No studies" ($A \leftarrow 0$),
$$\Delta = Y(1) - Y(0),$$
where $Y(t)$ is the potential outcome of intervening with $A\leftarrow a$. In particular, we are interested in the conditional average treatment effect (CATE),
$$\mathrm{CATE}(z) = \mathbb{E}[\Delta \mid Z=z]$$
where $Z \subseteq X$ is a given set of covariates. For this, we consider three main conditioning sets: 
1. $Z$ is the set of all pre-intervention covariates
2. $Z$ is the set of direct causes of $A$
3. $Z$ is a subset of covariates which is an invalid adjustment set. Specifically, $Z = (\mathrm{age}, \mathrm{education}, \mathrm{income\\_prev})$.

In addition, we seek to estimate the average treatment effect (ATE), $$\mathrm{ATE} = \mathbb{E}[\Delta]$$ using the first two conditioning sets above for adjustment. 

* *Note: The intervention variable ```studies``` is simulated with 4 values: Full-time studies, No studies, Evening course and Day course. In the Tasks above, samples with interventions other than Full-time studies or No studies can be discarded, or used for learning, depending on the estimator.*

### Evaluation

We measure the quality in estimates by the $R^2$, MSE, RMSE for CATE and the absolute error (AE) for ATE. Due to the complexity of the simulator, the CATE and ATE are not known as closed-form. 
Instead, we sample both counterfactual outcomes for a fixed of baseline subjects and compare their outcomes to each other. 

### File description

The main data set files are:
  * ```IncomeSCM-1.0.CATE_default.pkl``` (V1 simulator, default policy ($A$ observational), 50 000 samples, horizon T=5, seed=0)
  * ```IncomeSCM-1.0.CATE_no.pkl``` (V1 simulator, "No studies" policy ($A \leftarrow 0$), 50 000 samples, horizon T=5, seed=1)
  * ```IncomeSCM-1.0.CATE_full.pkl``` (V1 simulator, "Full-time studies" policy ($A \leftarrow 1$), 50 000 samples, horizon T=5, seed=1)
  * All three files are contained in [IncomeSCM-1.0.CATE.zip](samples/IncomeSCM-1.0.CATE.zip)

* **Training data**: The "default" policy data set represents observational data for causal effect estimators to learn from.
* **Evaluation data**: The "full" and "no" policy data sets represent samples observed under alternative interventions ($A \leftarrow 1$ and $A \leftarrow 0$, respectively).
  The out-of-sample quality of estimates of CATE and ATE can be estimated by using a model fit to the training data to predict (average) potential outcome for the subjects in the file representing each intervention and compare to the observed values. In Python, using the "S-learner" estimator implemented in the IncomeSCM package:
```python
import pandas as pd
import numpy as np
from income.estimators import S_learner

dobs = pd.read_pickle('samples/IncomeSCM-1.0.CATE_default.pkl')
d1 = pd.read_pickle('samples/IncomeSCM-1.0.CATE_no.pkl')
d0 = pd.read_pickle('samples/IncomeSCM-1.0.CATE_full.pkl')

model = S_learner(base_estimator=..., c_int='studies', c_out='income', c_adj=[...]).fit(dobs) # c_adj is the set of adjustment variables.
                                                                                              # base_estimator is any regression estimator. For the example to work
                                                                                              #   out of the box, it must handle categorical attributes in dobs[c_adj]
                                                                                              #   Alternatively, one-hot encoders can be used in e.g., a pipeline

yp1 = model.predict_outcomes(d1)
yp0 = model.predict_outcomes(d0)

cate_pred = yp1 - yp0
cate_true = d1['income'] - d0['income']
mse_cate = np.mean(np.square(cate_pred - cate_true))

ate_pred = np.mean(cate_pred)
ate_true = np.mean(cate_true)

ae_ate = np.abs(ate_pred - ate_true)
```
A real fitting and evaluation example is given in ```estimate.py```

## Using the simulator and estimators (IncomeSCM-1.0)

* IncomeSCM is written in Python 3 and based on the Scikit-learn package and the Adult dataset.
  
### Prerequisites

* To reproduce results or use the simulator, start by installing python modules ```pandas, numpy, scikit-learn, jupyter, matplotlib, yaml, xgboost```, for example in a virtual environment. Below, we list the versions used during development and testing. 
  ```
  pip install scikit-learn==1.4.1.post1 pandas==2.0.1 PyYAML==6.0 xgboost==2.0.0 matplotlib==3.7.1
  ```
* Download the IncomeSCM simulator
  ```
  git clone git@github.com:Healthy-AI/IncomeSCM.git
  ```

* The IncomeSCM simulator is fit to the [Adult dataset](http://archive.ics.uci.edu/dataset/2/adult) data set.
* To fit the simulator, run the python script ```fit.py``` in the repository folder
```
python fit.py [-c CONFIG_FILE]
```
* The default config file is configs/config_v1.yml
* To sample from the simulator, use the script ```sample.py```
```
python sample.py [-c CONFIG_FILE]
```
* This also uses the same default config file, which specifies which fitted model to use, how many samples are used, and from which (counterfactual) policy to sample. By default, 50 000 samples are generated from the "default" (observational) "full" and "no" policies.
* The samples are stored (by default) in ```./samples/[SAMPLE_FILE_NAME].pkl```. The file name is determined by the version labels specified in the config file.



# Papers using the data set 

<!--
# Lectures using the data set 

## DAT465 Lecture [2023]

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

### Coding in the demo

* Open [dat465_lecture_demo.ipynb](demos/dat465_lecture_demo.ipynb) in Jupyter in a Python environment with the prerequisites above
```bash
jupyter notebook   
```

## ProbAI 23 lecture [2023]

If you want to follow along in the notebook during the ProbAI lecture, you have two options: 
1. Clone this repository and open [probai_lecture_github.ipynb](demos/probai_lecture_github.ipynb) in Jupyter/Jupyter lab
2. Work in Colab from this [notebook](https://colab.research.google.com/drive/1jlEsSYcCDiqhamshxhkdQ703KKWaJHL9?usp=sharing)

The slides for the lecture can be found [here](demos/ProbAI_Causal_machine_learning.pdf).

**Installing prerequisites**

* IncomeSCM is written in Python 3 and based on the Scikit-learn package and the Adult dataset. 
* Start by installing python modules ```pandas, numpy, scikit-learn, jupyter, requests, matplotlib```

**Preparing the data files** 

You don't need to do this if you use the ProbAI notebook, the notebook does this automatically!

* Download the [Adult dataset](http://archive.ics.uci.edu/dataset/2/adult)
* Create a folder ``` data/income ``` in the IncomeSCM root folder
* Place the files ``` adult.data ```, ``` adult.names ``` and ``` adult.test ``` in ``` data/income ```

**Generating data**

* Run ``` python generate.py -n <number of samples> -T <length of horizon> ``` to fit the simulator and generate data
-->
