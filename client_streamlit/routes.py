import streamlit as st
import requests

# This module allows for the communication with the api

api_domain = 'http://35.173.161.134:5000/'

def post_customer_id(customer_id):

    """Checks if the User ID exists in the DataFrame"""

    return requests.post(api_domain + 'get_customer_id/', 
                          json={"customer_id": "{}".format(customer_id)}).json()

@st.cache()
def get_setup_infos():

    """Returns all features in the dataframe as well as all the categorical features only"""

    return requests.get(api_domain + 'get_setup_infos').json()

def predict_proba_customer(customer_id):

    """Returns the predicted proba of a given customer id"""

    return requests.post(api_domain + 'get_prediction_proba/', json={"customer_id": "{}".format(customer_id)}).json()

@st.cache()
def get_force_plot(customer_id):

    """ Returns a customer shap value with the forces of features in order to force
        plot it front end """
    
    return requests.post(api_domain + 'get_force_plot', json={"customer_id": "{}".format(customer_id)}).json()

def post_feature_customer_value(customer_id, feature_name):

    """Returns the feature value of a customer given an id and a feature"""

    return requests.post(api_domain + 'get_feature_customer_value', json={"customer_id": "{}".format(customer_id),
                                                                                   "feature_name": "{}".format(feature_name)}).json()

def get_group_value(feature, category, customer_category_value):

    """Returns a feature's group value filtered (or not) by a given (or not) category 
    with the value of a given (or not) customer's category value"""

    return requests.post(api_domain + 'get_group_value', json={"feature": "{}".format(feature),
                                                                "category": "{}".format(category),
                                                                "customer_category_value": "{}".format(customer_category_value)
                                                                }).json()