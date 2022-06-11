import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import streamlit.components.v1 as components
import plotly.express as px
import shap
import numpy as np
import routes

# Gives a title an an icon to the web page

st.set_page_config(
   page_title="Dashboard",
   page_icon = "./static/icon_pret_a_depenser.png"
)

# Gives a centered title with html component as natural markdown doesn't 
# allows centering.
# NB: as suggested use st.markdown property unsafe_allow_html=True would 
# allows tiers to inject code in our page hence the usage of component instead.

components.html("""<body style="margin: 0">
                        <h1 style="margin: 0;font-family: Source Sans Pro, sans-serif;
                        font-weight: 700; color: rgb(49, 51, 63);font-size: 2.45rem;text-align: center;">
                        <span>Dashboard</span>
                        </h1>
                    <body>""", 
                    width=None, height=50, scrolling=False)

# Given an id checks if it exists

typed_id = st.text_input('Enter a customer ID', value="", placeholder="You can try 100002 or 100003")
selected_id = routes.post_customer_id(typed_id)['customer_id']

# Loads features names and categorical feature names to display them in select box
with st.spinner('Select Box Loading...'):

    setup_infos = routes.get_setup_infos()
    selected_feature = st.selectbox('Choose a feature to explore:', setup_infos['all_features'])
    selected_category = st.selectbox('Choose a group to compare with :', setup_infos['all_categories'])

# If the id isn't correct (select_id) asks for correction
if selected_id == "":
    st.error('The selected ID isn\'t correct.')
    st.empty()
else: 

    # Retrieves and displays the predicted probability of our model

    with st.spinner('Computing Prediction...'):
        prediction = routes.predict_proba_customer(selected_id)

        if prediction['prediction'] >= 0.5:
            st.success('Loan Accepted ! Score : {}/100'.format(round(prediction['prediction']*100, 2)))
        else:
            st.error('Unfortunately the loan isn\'t accepted. Score : {}/100'.format(round(prediction['prediction']*100, 2)))

    # Retrieves the api computed shap value and feature forces to force plot
    # them
    with st.spinner('Loading Force Plot...'):
        # Retrieves the values
        get_force_plot_values = routes.get_force_plot(selected_id)
        # Loads values of the forces of features into a variable
        specific_shap_value = get_force_plot_values['specific_shap_value']
        # Loads features names into a variable
        specific_shap_value_features = get_force_plot_values['feature_names']
        # Plots the above variables
        force_plot = shap.force_plot(0, 
                                     np.array(specific_shap_value), 
                                     feature_names=specific_shap_value_features)
        # Creates a HTML component with get.js() in the head to be able to display
        # the force plot
        force_plot_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        st.write('About the features that impacted the most the above score :')
        components.html(force_plot_html)

    # Retrives and displays the customer data of the above given features through selectboexs
    with st.spinner('Retreiving cutomer data...'):
        # Retrives data
        feature_customer_value = routes.post_feature_customer_value(selected_id, selected_feature)
        category_customer_value = routes.post_feature_customer_value(selected_id, selected_category)
        feature_group_value = routes.get_group_value(selected_feature, selected_category, category_customer_value['feature_customer_value'])
        # Displays it in columns
        col_cust1, col_cust2 = st.columns(2)
        col_cust1.info('Customer\'s {} : {}'.format(re.sub('_', ' ', selected_feature.lower()), feature_customer_value['feature_customer_value']))
        col_cust2.info('Customer\'s {} : {}'.format(re.sub('_', ' ', selected_category.lower()), category_customer_value['feature_customer_value']))

    # If the explored feature (selected_feature) is numerical displays an expander
    # to display the group values (to display the value a pandas.describe() gives
    # -- this is done through the api) and displays the difference with the customer value

    if feature_group_value['feature_type'] == 'Numerical':
        # The expander is shaped with 3 rows of [3,3,2] columns each
        with st.expander('Group\'s metrics'):

            col1, col2, col3 = st.columns(3)

            col1.metric("count", feature_group_value['group_value']['count'])
            col2.metric("max", feature_group_value['group_value']['max'], 
                        feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['max'])
            col3.metric("min", feature_group_value['group_value']['min'], 
                        feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['min'])

            col6, col7, col8 = st.columns(3)

            col6.metric("25%", feature_group_value['group_value']['25%'], 
                        feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['25%'])
            col7.metric("50%", feature_group_value['group_value']['50%'], 
                        feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['50%'])
            col8.metric("75%", feature_group_value['group_value']['75%'], 
                        feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['75%'])

            col4, col5 = st.columns(2)

            col4.metric("mean", feature_group_value['group_value']['mean'], 
                        feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['mean'])
            col5.metric("std", round(feature_group_value['group_value']['std'], 2))

    # Two interactives plots are available : 
    # a boxplot if the main feature is a numerical one
    # a barplot if the main feature is a categorical one 
    with st.spinner('Visualisation Loading'):
        if feature_group_value['feature_type'] == 'Numerical':

            if feature_group_value['group_value']['count'] == 0:
                st.subheader('No Visualization available (NaNs)')
            else:
                # Feature is numerical => Boxplot (with striplot by its side to
                # see the id of customers and locate them)
                fig = px.box(
                            pd.DataFrame(
                                feature_group_value['values_list'], 
                                columns=['SK_ID_CURR',selected_feature]),
                            y=selected_feature,
                            hover_data=['SK_ID_CURR',selected_feature],
                            points="all",
                            labels={
                                    "variable": re.sub('_', ' ', selected_feature.lower())
                            },
                            title='{} representation inside {} category = {}'.format(re.sub('_', ' ', selected_feature.lower()),
                                                                            re.sub('_', ' ', selected_category.lower()), 
                                                                            category_customer_value['feature_customer_value']))
                fig.update_xaxes(tickvals=[""])
                fig.add_hline(y=feature_customer_value['feature_customer_value'],
                              annotation_text="You are here")
                st.plotly_chart(fig)
        else:

            if len(feature_group_value['group_value']) == 0:
                st.subheader('No Visualization available (NaNs)')
            else :
                # Feature is categorical => Barplot of the percentages of the classes
                fig = px.bar(
                        pd.DataFrame(data=list(dict(feature_group_value['group_value']).values()), 
                                     index = list(dict(feature_group_value['group_value']).keys()),
                            columns = ['Percentages']),
                        text_auto=True,
                        labels={"value": "Percentages",
                                "index": ""},
                        title='Representation among {} = {}'.format(re.sub('_', ' ', selected_category.lower()), 
                                                                 category_customer_value['feature_customer_value'])
                        )
                st.plotly_chart(fig)