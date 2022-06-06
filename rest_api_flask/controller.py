from flask import Flask, abort, request 
import pandas as pd
import joblib

app = Flask(__name__)

dataframe = pd.read_csv('./static/api_df_2.csv')

@app.route('/get_customer_id/', methods=['POST'])
def get_customer_id():

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

    categories_list = ['EVERY_CLIENTS'] + dataframe.select_dtypes('object').columns.tolist()

    return {'all_features':dataframe.columns.tolist()[1:],
            'all_categories':categories_list}

@app.route('/get_prediction/', methods=['POST'])
def get_prediction():

    request_data = request.get_json()

    user_data = dataframe[dataframe['SK_ID_CURR'] == int(request_data['customer_id'])].iloc[:,1:]

    if len(user_data) == 0:
        abort(404)
    else :

        model=open("./machine/test_light_gbm.pkl","rb")
        
        lgbm_model=joblib.load(model)

        customer_prediction = lgbm_model.predict(user_data).tolist()

        return {'prediction': customer_prediction}

@app.route('/get_prediction_proba/', methods=['POST'])
def get_prediction_proba():

    request_data = request.get_json()

    user_data = dataframe[dataframe['SK_ID_CURR'] == int(request_data['customer_id'])].iloc[:,1:]

    if len(user_data) == 0:
        abort(404)
    else :

        model=open("./machine/test_light_gbm.pkl","rb")
        
        lgbm_model=joblib.load(model)

        customer_prediction = lgbm_model.predict_proba(user_data).tolist()[0][0]

        return {'prediction': customer_prediction}

@app.route('/get_feature_customer_value', methods=['POST'])
def get_feature_customer_value():

    request_data = request.get_json()

    if request_data['feature_name'] == 'EVERY_CLIENTS':
        return {'feature_customer_value':'everybody'}
    else:
        user_feature_data = dataframe[dataframe['SK_ID_CURR'] == int(request_data['customer_id'])][request_data['feature_name']].values.tolist()[0]
        return {'feature_customer_value':user_feature_data}

@app.route('/get_group_value', methods=['POST'])
def get_group_value():

    request_data = request.get_json()

    feature = request_data['feature']
    category = request_data['category']
    customer_category_value = request_data['customer_category_value']

    # Numerical
    if dataframe[request_data['feature']].dtypes != object:
        # EVERY_CLIENTS
        if request_data['category'] == 'EVERY_CLIENTS':
            group_value = dict(dataframe[feature].describe())
            values_list = dataframe[feature].values.tolist()
            return {'feature_type': 'Numerical',
                    'group_value' : group_value,
                    'values_list': values_list}
        # SubGroup
        else:
            group_value = dict(dataframe[dataframe[category] == customer_category_value][feature].describe())
            values_list = dataframe[dataframe[category] == customer_category_value][feature].values.tolist()
            return {'feature_type': 'Numerical',
                    'group_value' : group_value,
                    'values_list': values_list}
    # Categorical
    else:
        # EVERY_CLIENTS
        if request_data['category'] == 'EVERY_CLIENTS':
            group_value = dict(dataframe[feature].value_counts(normalize=True).mul(100))
            return {'feature_type': 'Categorical',
                    'group_value' : group_value}
        # SubGroup
        else:
            group_value = dict(dataframe[dataframe[category] == customer_category_value][feature]\
                               .value_counts(dropna=False, normalize=True).mul(100))

            return {'feature_type': 'Categorical',
                    'group_value' : group_value}



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)