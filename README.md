# Implement a scoring model

  
### Overview: Classification of credit applicants to predict their creditworthiness. Implementation of an interactive dashboard that displays classification results of a given customer as well as some related statistics.

After cleaning the data (mainly with a kernel Kaggle that applies typical table joins, aggregations, and imputations), the EDA primarily shows that our targets are imbalanced (the minority class being the uncreditworthy clients). The data is rebalanced with undersampling and oversampling techniques such as SMOTE or Cluster Under Sampling. 
 
Four models are implemented (logistic regression, random forest, XGBoost, LightGBM), optimized (grid search), and compared with a 5 folds cross-validation (stratification included). The ROC-AUC metric is selected to compare those models. LightGBM model with a calibrated threshold is chosen, trained, and interpreted (with Shapley values).

The interactive dashboard has a Flask backend with the trained model and a Streamlit client with Plotly's visualization for interactivity. Both the backend and client are powered on AWS's EC2 instances.

One can find the Dashboard at http://18.233.144.105:8501/. 

### Structure

This project is divided into three main files : 

**client_streamlit** is the client dashboard coded in python with Streamlit package

**rest_api_flask** is the API where one can find the customer data (only a sample -- 6000 -- of customers here), the serialized trained model, and the client's business logic.

**notebooks_presentation_methodology** is a file where we gather the data science work that explain the path taken to select and train the used model.

###  Installation
Start with, 
 #####  On both instances (Client & API)

```

sudo apt update

sudo apt install python3 python3-pip

sudo apt-get install python3-venv subversion tmux

```

  After that,

#####  On the API instance

  

```

svn checkout https://github.com/RKL13/DS_OC_P7/trunk/rest_api_flask

cd rest_api_flask

```
Or
  

##### On the Client instance

  

```

svn checkout https://github.com/RKL13/DS_OC_P7/trunk/client_streamlit

cd client_streamlit

```

  Then, 

##### For both instances

  

```

chmod +x deploy.sh

tmux new -s session_1

./deploy.sh

ctrl-b, d

```
