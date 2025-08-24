# **Online Machine Learning Techniques on Software Data Streams For Automated Issue Assignment**
This repository contains the source code developed  for the diploma thesis "Applying Machine Learning Techniques on Software Data
Streams for Automated Issue Assignment". This repository provides instructions and a list of required libraries to reproduce the results and the final graphs presented inside the thesis.

### Setting up the libraries

All the following instructions are intended for use within a **Conda** environment. Please ensure you have an active Conda environment before proceeding.
You can download and install anaconda [here](https://www.anaconda.com/download). 

First, it is required to create a conda environment and install the **rust** programming language in it, this can be done by executing the following command `conda create -c conda-forge -n rustenv rust`.

We activate the enviroment we just created with this command `conda activate rustenv`.


Next, install all the required libraries and packages by executing the following commands:

Installing **Cython** `conda install cython`

Installing **River** `conda install git+https://github.com/online-ml/river --upgrade`

Installing **Numpy** `conda install numpy`

Installing **Pandas** `conda install pandas`

Installing **Matplotlib** `conda install matplotlib`

Installing **Pymong** `conda install pymongo`

Installing **Scikit-learn** `conda install scikit-learn` or get the latest version `conda install -c conda-forge scikit-learn`.

Installing **Scipy** `conda install -c conda-forge scipy`.

### Retrieving and Processing the Required Data.

You can download the dataset and find informations about it from [here](https://zenodo.org/records/14253918).

To access the data without downloading the whole dataset, you have to set the `mongo_URL` variable inside the file `properties.py`.

Following that, you can retrieve the data by executing the script `1_collect_data.py`.

Next, the data needs to be processed to extract the desired features. This can be done by running the script `2_process_data.py`.

### Experimental Results for Different Configurations

Initially, an offline batch learning model is compared to an online learning model by running the script `5.2_Batch_Vs_Online_learning.py`.  

Next, three different models are compared and evaluated as baseline models for the online implementation of AdaBoost by running the script `5.3_Naive-Bayes_Vs_Adaboost.py`.

After that, the contribution of different text features is evaluated by running the script `5.4_Contribution_Text_Features.py`.

Lastly, we optimize our model and generate the final results and graphs by running the script `5.5_optimizing_model.py`.

### Optional Code

You can plot the class distribution for every project by running the script `class_distribution.py`.

You can assess the impact of preprocessing for every project by running the script `project_overview.py`.

You can observe the assigned issues for each developer seperately by running the script `drift_assignees.py`.

You can evaluate the impact of resetting the classifier by running the script `Reset_Vs_No_Reset.py`.
