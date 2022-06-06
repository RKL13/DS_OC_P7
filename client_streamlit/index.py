import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import streamlit.components.v1 as components

st.set_page_config(
   page_title="Dashboard",
   page_icon = "./static/icon_pret_a_depenser.png"
)

def post_customer_id(customer_id):
    return requests.post('http://35.173.161.134:5000/get_customer_id/', 
                          json={"customer_id": "{}".format(customer_id)}).json()

@st.cache()
def get_setup_infos():
    return requests.get('http://35.173.161.134:5000/get_setup_infos').json()

def predict_proba_customer(customer_id):
    return requests.post('http://35.173.161.134:5000/get_prediction_proba/', json={"customer_id": "{}".format(customer_id)}).json()

def post_feature_customer_value(customer_id, feature_name):
    return requests.post('http://35.173.161.134:5000/get_feature_customer_value', json={"customer_id": "{}".format(customer_id),
                                                                                   "feature_name": "{}".format(feature_name)}).json()

def get_group_value(feature, category, customer_category_value):
    return requests.post('http://35.173.161.134:5000/get_group_value', json={"feature": "{}".format(feature),
                                                                        "category": "{}".format(category),
                                                                        "customer_category_value": "{}".format(customer_category_value)
                                                                        }).json()

st.components.v1.html("""<body style="margin: 0">
                            <h1 style="margin: 0;font-family: Source Sans Pro, sans-serif;
                            font-weight: 700; color: rgb(49, 51, 63);font-size: 2.45rem;text-align: center;">
                                <span>Dashboard</span>
                            </h1>
                        <body>""", 
                      width=None, height=50, scrolling=False)

typed_id = st.text_input('Enter a customer ID', value="", placeholder="You can try 100002 or 100003")
selected_id = post_customer_id(typed_id)['customer_id']

with st.spinner('Select Box Loading...'):

    setup_infos = get_setup_infos()
    selected_feature = st.selectbox('Choose a feature to explore:', setup_infos['all_features'])
    selected_category = st.selectbox('Choose a group to compare with :', setup_infos['all_categories'])

if selected_id == "":
    st.error('The selected ID isn\'t correct.')
    st.empty()
else: 
    with st.spinner('Computing Prediction...'):
        prediction = predict_proba_customer(selected_id)

        if prediction['prediction'] >= 0.5:
            st.success('Loan Accepted ! Score : {}/100'.format(round(prediction['prediction']*100, 2)))
        else:
            st.error('Unfortunately the loan isn\'t accepted. Score : {}/100'.format(round(prediction['prediction']*100, 2)))

    with st.spinner('Retreiving cutomer data...'):
        feature_customer_value = post_feature_customer_value(selected_id, selected_feature)
        category_customer_value = post_feature_customer_value(selected_id, selected_category)
        feature_group_value = get_group_value(selected_feature, selected_category, category_customer_value['feature_customer_value'])
        

    col_cust1, col_cust2 = st.columns(2)


    col_cust1.info('Customer\'s {} : {}'.format(re.sub('_', ' ', selected_feature.lower()), feature_customer_value['feature_customer_value']))

    col_cust2.info('Customer\'s {} : {}'.format(re.sub('_', ' ', selected_category.lower()), category_customer_value['feature_customer_value']))

    if feature_group_value['feature_type'] == 'Numerical':

        with st.expander('Group\'s metrics'):

            col1, col2, col3 = st.columns(3)

            col1.metric("count", feature_group_value['group_value']['count'])
            col2.metric("max", feature_group_value['group_value']['max'], feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['max'])
            col3.metric("min", feature_group_value['group_value']['min'], feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['min'])

            col6, col7, col8 = st.columns(3)

            col6.metric("25%", feature_group_value['group_value']['25%'], feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['25%'])
            col7.metric("50%", feature_group_value['group_value']['50%'], feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['50%'])
            col8.metric("75%", feature_group_value['group_value']['75%'], feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['75%'])

            col4, col5 = st.columns(2)

            col4.metric("mean", feature_group_value['group_value']['mean'], feature_customer_value['feature_customer_value'] - feature_group_value['group_value']['mean'])
            col5.metric("std", round(feature_group_value['group_value']['std'], 2))

    with st.spinner('Visualisation Loading'):
        if feature_group_value['feature_type'] == 'Numerical':

            if feature_group_value['group_value']['count'] == 0:
                st.subheader('No Visualization available (NaNs)')
            else:
                fig, ax = plt.subplots()
                sns.boxplot(data=feature_group_value['values_list'],
                            orient='h',
                            showmeans=True,
                            ax = ax)
                ax.axvline(feature_customer_value['feature_customer_value'], color='red')

                plt.title('{} representation inside {} category = {}'.format(re.sub('_', ' ', selected_feature.lower()),
                                                                            re.sub('_', ' ', selected_category.lower()), 
                                                                            category_customer_value['feature_customer_value']), 
                        pad = 20)

                st.pyplot(fig)
        else:

            if len(feature_group_value['group_value']) == 0:
                st.subheader('No Visualization available (NaNs)')
            else :
                fig, ax = plt.subplots()
                sns.barplot(data=pd.DataFrame(feature_group_value['group_value'].items()), x=0, y=1,
                            ax=ax)

                for bars in ax.containers:
                    ax.bar_label(bars, fmt='%.2f')

                ax.set_ylim((0,100))
                ax.set_ylabel('Percentages')
                ax.set_xlabel('Subcategories')

                plt.title('Representation among {} = {}'.format(re.sub('_', ' ', selected_category.lower()), 
                                                                category_customer_value['feature_customer_value']), 
                        pad = 20)

                if pd.DataFrame(feature_group_value['group_value'].items()).shape[0] > 4:
                    ax.tick_params(axis='x', rotation=90)

                st.pyplot(fig)