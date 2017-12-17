#!/usr/bin/python3 -B

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Data wrangling brought to you by the1owl
# https://www.kaggle.com/the1owl/surprise-me

data = {
    'tra':
    pd.read_csv('../input/air_visit_data.csv'),
    'as':
    pd.read_csv('../input/air_store_info.csv'),
    'hs':
    pd.read_csv('../input/hpg_store_info.csv'),
    'ar':
    pd.read_csv('../input/air_reserve.csv'),
    'hr':
    pd.read_csv('../input/hpg_reserve.csv'),
    'id':
    pd.read_csv('../input/store_id_relation.csv'),
    'tes':
    pd.read_csv('../input/sample_submission.csv'),
    'hol':
    pd.read_csv('../input/date_info.csv').rename(columns={
        'calendar_date': 'visit_date'
    })
}

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(
        lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df].groupby(
        ['air_store_id', 'visit_datetime'], as_index=False)[[
            'reserve_datetime_diff', 'reserve_visitors'
        ]].sum().rename(columns={
            'visit_datetime': 'visit_date'
        })
    print(data[df].head())

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(
    lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(
    lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat(
    [
        pd.DataFrame({
            'air_store_id': unique_stores,
            'dow': [i] * len(unique_stores)
        }) for i in range(7)
    ],
    axis=0,
    ignore_index=True).reset_index(drop=True)

#sure it can be compressed...
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].min().rename(columns={
        'visitors': 'min_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].mean().rename(columns={
        'visitors': 'mean_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].median().rename(columns={
        'visitors': 'median_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].max().rename(columns={
        'visitors': 'max_visitors'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
tmp = data['tra'].groupby(
    ['air_store_id', 'dow'],
    as_index=False)['visitors'].count().rename(columns={
        'visitors': 'count_observations'
    })
stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])

train = pd.merge(data['tra'], stores, how='left', on=['air_store_id', 'dow'])
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id', 'dow'])

for df in ['ar', 'hr']:
    train = pd.merge(
        train, data[df], how='left', on=['air_store_id', 'visit_date'])
    test = pd.merge(
        test, data[df], how='left', on=['air_store_id', 'visit_date'])

col = [
    c for c in train
    if c not in ['id', 'air_store_id', 'visit_date', 'visitors']
]
train = train.fillna(-1)
test = test.fillna(-1)

# XGB starter template borrowed from @anokas
# https://www.kaggle.com/anokas/simple-xgboost-starter-0-0655

print('Binding to float32')

for c, dtype in zip(train.columns, train.dtypes):
    if dtype == np.float64:
        train[c] = train[c].astype(np.float32)

for c, dtype in zip(test.columns, test.dtypes):
    if dtype == np.float64:
        test[c] = test[c].astype(np.float32)

train_x = train.drop(['air_store_id', 'visit_date', 'visitors'], axis=1)
train_y = np.log1p(train['visitors'].values)
print(train_x.shape, train_y.shape)
test_x = test.drop(['id', 'air_store_id', 'visit_date', 'visitors'], axis=1)

# parameter tuning of xgboost
# start from default setting
boost_params = {'eval_metric': 'rmse'}
xgb0 = xgb.XGBRegressor(
    max_depth=7,
    learning_rate=0.04,
    n_estimators=10000,
    objective='reg:linear',
    gamma=0,
    min_child_weight=1,
    subsample=1,
    colsample_bytree=1,
    scale_pos_weight=1,
    seed=27,
    **boost_params)

xgb0.fit(train_x, train_y)
predict_y = xgb0.predict(test_x)
test['visitors'] = np.expm1(predict_y)
test[['id', 'visitors']].to_csv(
    'xgb0_submission.csv', index=False, float_format='%.3f')  # LB0.500

# Grid seach on subsample and max_features
param_test1 = {
    'max_depth': range(3, 20, 1),
    'min_child_weight': range(2, 8, 1)
}

gsearch1 = GridSearchCV(
    estimator=xgb.XGBRegressor(
        max_depth=7,
        learning_rate=0.04,
        n_estimators=10000,
        gamma=0,
        objective='reg:linear',
        min_child_weight=1,
        subsample=1,
        colsample_bytree=1,
        scale_pos_weight=1,
        seed=27,
        **boost_params),
    param_grid=param_test1,
    verbose=2,
    iid=False,
    cv=5)
gsearch1.fit(train_x, train_y)

print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

predict_y = gsearch1.predict(test_x)
test['visitors'] = np.expm1(predict_y)
test[['id', 'visitors']].to_csv(
    'xgb1_submission.csv', index=False, float_format='%.3f')  # LB0.513

#Grid seach on gamma
param_test2 = {'gamma': [i / 100.0 for i in range(0, 100, 5)]}

gsearch2 = GridSearchCV(
    estimator=xgb.XGBRegressor(
        max_depth=gsearch1.best_params_['max_depth'],
        learning_rate=0.04,
        n_estimators=10000,
        gamma=0,
        objective='reg:linear',
        min_child_weight=gsearch1.best_params_['min_child_weight'],
        subsample=1,
        colsample_bytree=1,
        scale_pos_weight=1,
        seed=27,
        **boost_params),
    param_grid=param_test2,
    verbose=2,
    iid=False,
    cv=5)
gsearch2.fit(train_x, train_y)

print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)

predict_y = gsearch2.predict(test_x)
test['visitors'] = np.expm1(predict_y)
test[['id', 'visitors']].to_csv(
    'xgb2_submission.csv', index=False, float_format='%.3f')  # LB0.512

#Grid seach on subsample, colsample_bytree
param_test3 = {
    'subsample': [i / 100.0 for i in range(6, 100, 5)],
    'colsample_bytree': [i / 100.0 for i in range(20, 100)]
}

gsearch3 = GridSearchCV(
    estimator=xgb.XGBRegressor(
        max_depth=gsearch1.best_params_['max_depth'],
        learning_rate=0.04,
        n_estimators=10000,
        gamma=gsearch2.best_params_['gamma'],
        min_child_weight=gsearch1.best_params_['min_child_weight'],
        objective='reg:linear',
        subsample=1,
        colsample_bytree=1,
        scale_pos_weight=1,
        seed=27,
        **boost_params),
    param_grid=param_test3,
    verbose=2,
    iid=False,
    cv=5)
gsearch3.fit(train_x, train_y)

print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)

predict_y = gsearch3.predict(test_x)
test['visitors'] = np.expm1(predict_y)
test[['id', 'visitors']].to_csv(
    'xgb3_submission.csv', index=False, float_format='%.3f')  # LB0.512

#Grid seach on reg_alpha
param_test4 = {'reg_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]}

gsearch4 = GridSearchCV(
    estimator=xgb.XGBRegressor(
        max_depth=gsearch1.best_params_['max_depth'],
        learning_rate=0.04,
        n_estimators=10000,
        gamma=gsearch2.best_params_['gamma'],
        min_child_weight=gsearch1.best_params_['min_child_weight'],
        objective='reg:linear',
        subsample=gsearch3.best_params_['subsample'],
        colsample_bytree=gsearch3.best_params_['colsample_bytree'],
        scale_pos_weight=1,
        reg_alpha=0.005,
        seed=27,
        **boost_params),
    param_grid=param_test4,
    verbose=2,
    iid=False,
    cv=5)
gsearch4.fit(train_x, train_y)

print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

predict_y = gsearch4.predict(test_x)
test['visitors'] = np.expm1(predict_y)
test[['id', 'visitors']].to_csv(
    'xgb4_submission.csv', index=False, float_format='%.3f')  # LB0.512

# recalculate boosting round w/ reduced lr and increased estimators
xgb5 = xgb.XGBRegressor(
    max_depth=gsearch1.best_params_['max_depth'],
    learning_rate=0.001,
    n_estimators=5000,
    gamma=gsearch2.best_params_['gamma'],
    min_child_weight=gsearch1.best_params_['min_child_weight'],
    objective='reg:linear',
    subsample=gsearch3.best_params_['subsample'],
    colsample_bytree=gsearch3.best_params_['colsample_bytree'],
    scale_pos_weight=1,
    reg_alpha=gsearch4.best_params_['reg_alpha'],
    seed=27,
    **boost_params)

xgb5.fit(train_x, train_y)
predict_y = xgb5.predict(test_x)
test['visitors'] = np.expm1(predict_y)
test[['id', 'visitors']].to_csv(
    'xgb5_submission.csv', index=False, float_format='%.3f')  # LB0.512
