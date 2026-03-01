import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb
import sklearn
import warnings
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer, FunctionTransformer, OrdinalEncoder, StandardScaler
from feature_engine.encoding import RareLabelEncoder, MeanEncoder, CountFrequencyEncoder
from feature_engine.datetime import DatetimeFeatures
from feature_engine.outliers import Winsorizer
from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.ensemble import RandomForestRegressor
sklearn.set_config(transform_output="pandas")
warnings.filterwarnings('ignore')
import streamlit as st

def is_north(X):
    north_cities = ['Delhi','Kolkata','Mumbai']
    cols = X.columns.to_list()
    return (X
        .assign(
        **{
            f"{col}_is_north": X.loc[:,col].isin(north_cities).astype('int') for col in cols
        }
        )
        .drop(columns=cols)
    )
def get_part_of_day(X, early_morning=4, morning=8, noon=12, afternoon=16, evening=19, night=22, late_night=1):
    cols = X.columns.to_list()
    temp_time_df = X.assign(**{
        col: pd.to_datetime(X.loc[:,col]).dt.hour for col in cols
    })
    return temp_time_df.assign(**{
        col: np.select(
            [
                temp_time_df.loc[:,col].between(early_morning,morning,inclusive='left'),
                temp_time_df.loc[:,col].between(morning,noon,inclusive='left'),
                temp_time_df.loc[:,col].between(noon,afternoon,inclusive='left'),
                temp_time_df.loc[:,col].between(afternoon,evening,inclusive='left'),
                temp_time_df.loc[:,col].between(evening,night,inclusive='left'),
                temp_time_df.loc[:,col].between(night,late_night,inclusive='left'),
            ],
            ['early_morning','morning','noon','afternoon','evening','night'],
            default='late_night'
        ) for col in cols
    })

def duration_category(X, quick=0, medium=120, long=500):
    return (X.assign(duration_cat=np.select(
        [X.duration.between(quick,medium,inclusive='left'), 
        X.duration.between(medium, long, inclusive='left')],
        ['quick','medium'],
        default='long'
        )).drop(columns="duration"))

def is_direct(X):
    return X.assign(direct_flight=X.total_stops.eq(0)).astype(int)


def save_preprocessor(train):
    airline_transformer = Pipeline(
    steps=[
        ('simple_imputer', SimpleImputer(strategy='most_frequent')),
        ('group_rare_labels', RareLabelEncoder(tol=0.01,n_categories=3,replace_with='Other',missing_values='ignore')),
        ('one_hot_encoder',OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
    ]
    )

    features_to_extract = ["month","week","day_of_week","day_of_year"]
    date_of_journey_transformer = Pipeline(
        steps=[
            ('dt',DatetimeFeatures(features_to_extract=features_to_extract,yearfirst=True,format='mixed')),
            ('scaler',MinMaxScaler())
        ]
    )

    city_transformer = Pipeline(
        steps=[
            ('group_rare', RareLabelEncoder(tol=0.1, n_categories=3, replace_with='Other')),
            ('encoder', MeanEncoder()),
            ('scaler', PowerTransformer(standardize=True))
        ]
    )

    location_transformer = FeatureUnion(transformer_list=[
        ("step_1",city_transformer),
        ("step_2",FunctionTransformer(func=is_north))
    ])

    time_pipe_1 = Pipeline(
        steps=[
            ('time_feat',DatetimeFeatures(features_to_extract=['hour','minute'])),
            ('scaler',MinMaxScaler())
        ]
    )


    time_pipe_2 = Pipeline(steps=[
        ("part_of_the_day_trans", FunctionTransformer(func=get_part_of_day)),
        ("count_enco", CountFrequencyEncoder()),
        ("scaler", MinMaxScaler())
    ])

    time_transformer = FeatureUnion([
        ('step_1', time_pipe_1,),
        ('step_2',time_pipe_2)
    ])


    duration_pipe_1 = Pipeline(
        steps=[
            ("dur_cat",FunctionTransformer(func=duration_category)),
            ("ecoder", OrdinalEncoder(categories=[["quick", "medium", "long"]]))
        ]
    )
    duration_pipe_2 = Pipeline(steps=[
        ('outliers',Winsorizer(capping_method='iqr',fold=1.5)),
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    duration_trans = FeatureUnion([
        ("duration_pipe1",duration_pipe_2),
        ("duration_pipe2",duration_pipe_1)
    ])

    total_stops_trans = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('direct_check', FunctionTransformer(func=is_direct))
    ])

    add_info_trans = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("rare_label", RareLabelEncoder(tol=0.1, n_categories=3, replace_with='other')),
        ("encoder", OneHotEncoder(sparse_output=False))
    ])

    col_transformer = ColumnTransformer(
    [
        ('airline_trans',airline_transformer,['airline']),
        ('doj_trans',date_of_journey_transformer,['date_of_journey']),
        ('loc_trans',location_transformer,['source','destination']),
        ('time_trans',time_transformer,['dep_time','arrival_time']),
        ('dur_trans',duration_trans,['duration']),
        ('total_stops_trans', total_stops_trans,['total_stops']),
        ('add_info_trans', add_info_trans,['additional_info'])
    ],
    remainder='passthrough'
    )
    
    estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

    selection_trans = SelectBySingleFeaturePerformance(estimator, scoring='r2', threshold=0.1)

    preprocessor = Pipeline(steps=[
    ("columns_transform", col_transformer),
    ("feature_selction", selection_trans)
    ])
    preprocessor.fit(train.drop(columns="price"),train.price.copy())
    joblib.dump(preprocessor,"preprocessor.joblib")
    return True

def load_preprocessor():
    return joblib.load("preprocessor.joblib")

def load_model():
    with open("xgboost-model", "rb") as f:
        best_model = pickle.load(f)
    return best_model


if __name__ == "__main__":
    train_df = pd.read_csv('data/train_data.csv')
    save_preprocessor(train_df)

    st.set_page_config(
        page_title='Flight Price predictor',
        layout='wide'
    )

    st.title('Flight Price Predictor Using AWS')

    airline = st.selectbox("Airline:",options=train_df.airline.unique())
    doj = st.date_input("Date of Journey:")
    source = st.selectbox("Source", options=train_df.source.unique())
    destination = st.selectbox("Destination", options=train_df.destination.unique())
    dep_time = st.time_input("Departure Time:")
    arr_time = st.time_input("Arrival Time:")
    duration = st.number_input("Duration", step=1)
    total_stops = st.number_input("Total Stops:", step=1, min_value=0)
    additional_info = st.selectbox("Additional_Info:", options=train_df.additional_info.unique())

    pred_df = pd.DataFrame(dict(
        airline = [airline],
        date_of_journey = [doj],
        source = [source],
        destination = [destination],
        dep_time = [dep_time],
        arrival_time = [arr_time],
        duration = [duration],
        total_stops = [total_stops],
        additional_info = [additional_info]
    )).astype({col:"str" for col in ['date_of_journey','dep_time','arrival_time']})


    if st.button('Predict'):
        preprocessor = load_preprocessor()
        pred_df_pre = preprocessor.transform(pred_df)

        model = load_model()
        pred_df_pre_xgb = xgb.DMatrix(pred_df_pre)
        price = model.predict(pred_df_pre_xgb)[0]

        st.info(f"The price of the flight will be {price:,.0f} INR.")
