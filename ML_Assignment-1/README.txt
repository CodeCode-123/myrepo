# Supervised Learning -- Assignment-1 (CS 7641 Machine Learning)
# Name: Shen Tang, GTaccount: stang311

This project was to explore Decision Tree, Neural Networks, Boosting, Support Vector Machines, and K-Nearest Neighbors algorithms in supervised learning. This is the instruction to install packages and run the code on your local machine of Assignment-1.  

Please find this project on Github:
ML_Assignment-1
Please find the Github linker of this project from the README.txt submitted to Canvas. 
Please download and save the project folder "ML_Assignment-1" locally and run the code with the following instructions. 

In the "ML_Assignment-1" folder, you can find:
1. old.arff (Data of "Phishing-web" dataset)
2. page-blocks.data (Data of "Page-blocks" dataset)
3. page-blocks.ipynb (Jupyter notebook file of Machine Learning of "Page-blocks" dataset, make sure that "page-blocks.data" and "page-blocks.names" are in the same folder of this file)
4. page-blocks.names (Data of "Page-blocks" dataset)
5. Phishing.ipynb (Jupyter notebook file of Machine Learning of "Phishing-web" dataset, make sure that "old.arff" is in the same folder of this file)
6. README.txt (Please find the Github linker of this project from the README.txt submitted to Canvas, but not in this file)

This project required to have python3, Jupyter Notebook, pandas, numpy, scikit-learn, matplotlib, seaborn installed. Please find the installation instruction at the end of this file. If you have already installed all the required packages, you can skip the installation. 

You can find the websites of packages below:
[python3](https://www.python.org/downloads/)
[Jupyter notebook](https://jupyter.org)
[pandas](https://pandas.pydata.org)
[numpy](https://numpy.org)
[scikit-learn](https://scikit-learn.org/stable/)
[matplotlib](https://matplotlib.org/stable/index.html)
[seaborn](https://seaborn.pydata.org)

The versions of packages I've installed were as below:
Python 3.8.0

Jupyter related:
jupyter core       4.7.0
jupyter-notebook   6.1.6
qtconsole          not installed
ipython            7.19.0
ipykernel          5.4.2
jupyter client     6.1.11
jupyter lab        3.0.3
nbconvert          6.0.7
ipywidgets         not installed
nbformat           5.0.8
traitlets          5.0.5

pandas        1.2.0
numpy         1.19.1
scikit-learn  0.24.0
matplotlib    3.3.3
seaborn       0.11.1
pip           20.2.3


# Commands
Open the terminal
Type the commands after $ in the terminal
Click return and run


# Datasets 
There are two datasets from UCI in this project for machine learning analysis, please find the linkers of these two datasets below:

[Phishing-web](https://archive.ics.uci.edu/ml/machine-learning-databases/00327/) 
[Page-blocks](https://archive.ics.uci.edu/ml/machine-learning-databases/page-blocks/)

The ".old.arff" of Phishing-web and "page-blocks.data.Z" and "page-blocks.names" of Page-blocks were downloaded and saved or unpacked and saved locally as "old.arff", "page-blocks.data", and "page-blocks.names" for data processing and analysis. They are all in the project folder.

# Run the code
1. Download and save the project folder "ML_Assignment-1" locally
2. Open a terminal and type the command of "$ jupyter notebook"
3. On the opened localhost website, select "Phishing.ipynb" in "ML_Assignment-1" under "Files" by double-clicking it. On the same website, select "Page-blocks.ipynb" in "ML_Assignment-1" under "Files" by double-clicking it. A new localhost website of "Phishing - Jupyter Notebook" and a new localhost website of "Page-blocks - Jupyter Notebook" were opened, and you can check the current results of each step. 
4. On the opened localhost website, select "Page-blocks.ipynb" by double-clicking it, and a new localhost website of "Page-blocks.ipynb" was opened, and you can check the current results of each step. 
5. On either the "Phishing - Jupyter Notebook" or "Page-blocks - Jupyter Notebook" website, please run the cell from the top to "# plot learning curve" (Cell 13 and 11 of these two programs, respectively) for data pre-processing.
6. If you'd like to run the code one by one, select the particular cell in the notebook, and from the menu bar select "Cell --> Run Cells", the results of this cell will show below.
7. If you'd like to run all the codes, from the menu bar select "Cell --> Run All".

*Notes: 
1). All the savefig commands were commented out to prevent path error, and the plots were visualized below each cell. 
2). The Precision, Recall, f1, and normalized confusion matrixes in the report were printed out on the top of each ROC plot in this notebook. They were manually listed in the table of a PowerPoint file and plotted in the report. 
3). The Train and cross-validation (CV) accuracies were printed on the top of each learning curve in this notebook. They were manually listed in the table of a PowerPoint file and plotted in the report.  
4). The average precision (AP) values were printed on the top of the PRAUC curve in this notebook. They were manually listed in the table of a PowerPoint file and plotted in the report.
5). The "cv_scores_mean" under the hyperparameter (HP) tuning was the list of the CV accuracies of the tuned hyperparameter. The optimal value of HP can be identified by finding the maximal CV accuracy.    
6). The accuracy, precision, recall, f1 score, confusion matrix were highly reproducible. There might be slight differences in the fitting time and fitting time plots in the re-run. However, these small differences would not significantly influence the conclusion.   


# References 
Code in this project from the websites
[process .arff type files](https://discuss.analyticsvidhya.com/t/loading-arff-type-files-in-python/27419)
[Display count number on the plot](https://stackoverflow.com/questions/42159250/how-to-show-the-count-values-on-the-top-of-a-bar-in-a-countplot)
[pd DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
[DataFrame Processing](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)
[train_test_split stratify](https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn)
[Plot learning curve](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html)
[Plot learning curve-2](https://stackoverflow.com/questions/48018203/what-is-the-difference-between-knn-score-and-accuracy-metrics-in-knn-sk-learn)
[DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
[MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
[AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
[SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
[KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
[ROC plots](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html)
[PRAUC](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py)
[Confusion metrics](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
[Model Complexity Curve plot, HP tuning](https://towardsdatascience.com/how-to-find-decision-tree-depth-via-cross-validation-2bf143f0f3d6)
[GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
[precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
[precision_recall_fscore_support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
[fitting time](https://stackoverflow.com/questions/61586009/timing-of-model-testing-and-training-of-a-decision-tree-classifier)
[plot bar-1](https://www.tutorialspoint.com/matplotlib/matplotlib_bar_plot.htm)
[plot bar-2](https://numpy.org/devdocs/reference/generated/numpy.around.html)



########################

# Installation 
Make sure that you have installed python3, and I had Python 3.8.0 installed.
[Python 3.8.0 installation](https://www.python.org/downloads/release/python-380/)
For the Mac user, please download and install "macOS 64-bit installer" in the Files
For the Window user, please download and install "Windows x86-64 executable installer" in the Files

Make sure that you can install the software by pip, usually, it was co-installed with Python 3
[Instruction of checking pip version](https://pip.pypa.io/en/stable/installing/)
$ python3 -m pip --version

[Instruction of installing JupyterLab](https://jupyter.org/install)
$ pip install jupyterlab

[Instruction of installing Jupyter Notebook](https://jupyter.org/install)
$ pip install notebook

[Instruction of installing pandas](https://pypi.org/project/pandas/)
$ pip install pandas

[Instruction of installing numpy](https://numpy.org/install/)       
$ pip install numpy

[Instruction of installing scikit-learn](https://scikit-learn.org/stable/install.html)
$ pip install -U scikit-learn

[Instruction of installing matplotlib](https://pypi.org/project/matplotlib/)
$ pip install matplotlib

[Instruction of installing seaborn](https://pypi.org/project/seaborn/)       
$ pip install seaborn

###################