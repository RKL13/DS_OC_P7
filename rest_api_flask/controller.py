from flask import Flask, abort, request, jsonify
import pandas as pd
import joblib
import shap
import re
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

app = Flask(__name__)

# Loads the Dataframe

dataframe = pd.read_csv('./static/app_6000.csv')

# Loads the Model
model=open("./machine/lightgbm_all_data.pkl","rb")
lgbm_model=joblib.load(model)

# Loads and intits the shap TreeExplainer on the model

tree_explainer = shap.TreeExplainer(lgbm_model[-1])

# Computes shap values at the server init because these won't change until
# CSV is changed

# Transforms dataframe

X_transformed = lgbm_model[:-1].transform(dataframe.iloc[:,1:])

# Retreives features names after transformation
columns_names_out = lgbm_model[:-1].get_feature_names_out()

# Removes added Pipelines particules

columns_names_out_cleaned = \
            [re.sub('(Categorical_pipeline__)|(Numerical_pipeline__)', '', i) 
            for i in columns_names_out]

# Computes shap values for all transformed individuals
# N.B : One can reduce the sample size using shap.sample if it is too large

all_shap_values = tree_explainer.shap_values(X_transformed)

# Creates a matplotlib figure 

shap.summary_plot(all_shap_values, 
                  feature_names=columns_names_out_cleaned, 
                  show=False)

plt.tight_layout()
plt.savefig('./static/summary_plot.png')

# Routing
 
@app.route('/get_customer_id/', methods=['POST'])
def get_customer_id():

    """Checks if the User ID exists in the DataFrame"""

    request_data = request.get_json()

    customer_id = request_data['customer_id']
    
    try : 
        if len(dataframe[dataframe['SK_ID_CURR'] == int(customer_id)]) == 0:
            check = ""
        else:
            check = customer_id
    except:
        check =""

    return {"customer_id" : check}

@app.route('/get_setup_infos')
def get_setup_infos():

    """Returns all features in the dataframe as well as all the categorical features only"""

    categories_list = ['EVERY_CLIENTS'] + dataframe.select_dtypes('object').columns.tolist()

    return {'all_features':dataframe.columns.tolist()[1:],
            'all_categories':categories_list}

@app.route('/get_prediction_proba/', methods=['POST'])
def get_prediction_proba():

    """Returns the predicted proba of a given customer id"""

    request_data = request.get_json()

    # Retrieves the row of a user given an id
    user_data = dataframe[dataframe['SK_ID_CURR'] == int(request_data['customer_id'])].iloc[:,1:]

    if len(user_data) == 0:
        abort(404)
    else :
        # Uses the model to predict proba
        customer_prediction = lgbm_model.predict_proba(user_data).tolist()[0][0]

        return {'prediction': customer_prediction}
 
@app.route('/get_feature_customer_value', methods=['POST'])
def get_feature_customer_value():

    """Returns the feature value of a customer given an id and a feature"""

    request_data = request.get_json()

    if request_data['feature_name'] == 'EVERY_CLIENTS':
        return {'feature_customer_value':'everybody'}
    else:
        user_feature_data = dataframe[dataframe['SK_ID_CURR'] == int(request_data['customer_id'])][request_data['feature_name']].values.tolist()[0]
        return {'feature_customer_value':user_feature_data}

@app.route('/get_group_value', methods=['POST'])
def get_group_value():

    """Returns the values of a given features to make comparison with a customer value.
       If the group feature is numerical all the values of the feature are returned
       to plot it front end. 
       If the group feature is categorical only value_counts() are returned to 
       bar plot it front end.
       In both cases (Numerical, Categorical) the values can be filtered by another
       given subgroup which is always a categorical one and with the value of the 
       compared user."""

    request_data = request.get_json()

    feature = request_data['feature']
    category = request_data['category']
    customer_category_value = request_data['customer_category_value']

    # Numerical
    if dataframe[request_data['feature']].dtypes != object:
        # EVERY_CLIENTS
        if request_data['category'] == 'EVERY_CLIENTS':
            group_value = dict(dataframe[feature].describe())
            values_list = dataframe[['SK_ID_CURR', feature]].values.tolist()
            return {'feature_type': 'Numerical',
                    'group_value' : group_value,
                    'values_list': values_list}
        # SubGroup
        else:
            group_value = dict(dataframe[dataframe[category] == customer_category_value][feature].describe())
            values_list = dataframe[dataframe[category] == customer_category_value][['SK_ID_CURR', feature]].values.tolist()
            return {'feature_type': 'Numerical',
                    'group_value' : group_value,
                    'values_list': values_list}
    # Categorical
    else:
        # EVERY_CLIENTS
        if request_data['category'] == 'EVERY_CLIENTS':
            group_value = dataframe[feature].value_counts(normalize=True).mul(100)
            group_value.index = group_value.index.fillna('Missing values')
            group_value = dict(group_value)
            return {'feature_type': 'Categorical',
                    'group_value' : group_value}
        # SubGroup
        else:
            group_value = dataframe[dataframe[category] == customer_category_value][feature]\
                               .value_counts(dropna=False, normalize=True).mul(100)

            group_value.index = group_value.index.fillna('Missing values')

            group_value = dict(group_value)

            return {'feature_type': 'Categorical',
                    'group_value' : group_value}

@app.route('/get_force_plot', methods=['POST'])
def get_force_plot():

    """ Returns a customer shap value with the forces of features in order to force
        plot it front end """

    request_data = request.get_json() 

    customer_row = dataframe[dataframe['SK_ID_CURR'] == int(request_data['customer_id'])].copy()
    
    if len(customer_row) == 0:
        abort(404)
    else :
        # Removes SK_ID
        customer_row = customer_row[[i for i in customer_row.columns if i not in ['SK_ID_CURR']]]

        # Applies Pipeline Transformations
        row_transformed = lgbm_model[:-1].transform(customer_row)

        # Retrives the column names
        columns_names_out = lgbm_model[:-1].get_feature_names_out()
        # Cleans the retrieved column names
        columns_names_out_cleaned = \
            [re.sub('(Categorical_pipeline__)|(Numerical_pipeline__)', '', i) 
            for i in columns_names_out]

        # Computes a local SHAP value
        specific_shap_value = tree_explainer.shap_values(row_transformed)

        return {'specific_shap_value': specific_shap_value[0][0].tolist(),
                'feature_names': columns_names_out_cleaned,
                'expected_value_shap': tree_explainer.expected_value[0]}

@app.route('/get_summary_plot')
def get_summary_plot():

    """ Returns base 64 encoded summary plot """

    img = Image.open("./static/summary_plot.png", mode='r')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')


    return jsonify({'summary_plot': encoded_img})


# Runs the app on port :5000
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)