import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

def featureModify():

    all_data = pd.read_csv('train_V2.csv')
    all_data = all_data[all_data['winPlacePerc'].notnull()]

    # rank as percent
    match = all_data.groupby('matchId')
    all_data['killsPerc'] = match['kills'].rank(pct=True).values
    all_data['killPlacePerc'] = match['killPlace'].rank(pct=True).values
    all_data['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values
    all_data['walkPerc_killsPerc'] = all_data['walkDistancePerc'] / all_data['killsPerc']

    # distance
    all_data['_totalDistance'] = all_data['rideDistance'] + all_data['walkDistance'] + all_data['swimDistance']

    # new features
    all_data['_healthItems'] = all_data['heals'] + all_data['boosts']
    all_data['_headshotKillRate'] = all_data['headshotKills'] / all_data['kills']
    all_data['_killPlaceOverMaxPlace'] = all_data['killPlace'] / all_data['maxPlace']
    all_data['_killsOverWalkDistance'] = all_data['kills'] / all_data['walkDistance']



    # drop features
    all_data.drop(['boosts','heals','killStreaks','DBNOs'], axis=1, inplace=True)
    all_data.drop(['headshotKills','roadKills','vehicleDestroys'], axis=1, inplace=True)
    all_data.drop(['rideDistance','swimDistance','matchDuration'], axis=1, inplace=True)
    
    all_data.loc[(all_data['rankPoints']==-1), 'rankPoints'] = 0

    all_data = all_data.replace([np.inf, np.NINF, np.nan], 0)

    # grouping
    match = all_data.groupby(['matchId'])
    group = all_data.groupby(['matchId','groupId'])

    # aggregate features
    agg_col = list(all_data.columns)
    exclude_agg_col = ['Id','matchId','groupId','matchType','maxPlace','numGroups','winPlacePerc']
    for c in exclude_agg_col:
        agg_col.remove(c)

    
    # personal rank 
    all_data = all_data.join(all_data.groupby('matchId')[agg_col].rank(ascending=False).add_suffix('_rankPlace').astype(int))


    # group mean data and rank in perc
    meanData = group[agg_col].agg('mean').replace([np.inf, np.NINF,np.nan], 0)
    meanDataRank = meanData.groupby('matchId')[agg_col].rank(pct=True).reset_index()
    all_data = pd.merge(all_data, meanData.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
    del meanData
    gc.collect()
    all_data = pd.merge(all_data, meanDataRank, suffixes=["", "_meanRank"], how='left', on=['matchId', 'groupId'])
    del meanDataRank
    gc.collect()

    # group std data and rank in perc
    stdData = group[agg_col].agg('std').replace([np.inf, np.NINF,np.nan], 0)
    stdDataRank = stdData.groupby('matchId')[agg_col].rank(pct=True).reset_index()
    all_data = pd.merge(all_data, stdData.reset_index(), suffixes=["", "_std"], how='left', on=['matchId', 'groupId'])
    del stdData
    gc.collect()
    all_data = pd.merge(all_data, stdDataRank, suffixes=["", "_stdRank"], how='left', on=['matchId', 'groupId'])
    del stdDataRank
    gc.collect()


    # group max data and rank in perc
    maxData = group[agg_col].agg('max')
    maxDataRank = maxData.groupby('matchId')[agg_col].rank(pct=True).reset_index()
    all_data = pd.merge(all_data, maxData.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
    del maxData
    gc.collect()
    all_data = pd.merge(all_data, maxDataRank, suffixes=["", "_maxRank"], how='left', on=['matchId', 'groupId'])
    del maxDataRank
    gc.collect()

    # group min data and rank in perc
    minData = group[agg_col].agg('min')
    minDataRank = minData.groupby('matchId')[agg_col].rank(pct=True).reset_index()
    all_data = pd.merge(all_data, minData.reset_index(), suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
    del minData
    gc.collect()
    all_data = pd.merge(all_data, minDataRank, suffixes=["", "_minRank"], how='left', on=['matchId', 'groupId'])
    del minDataRank
    gc.collect()

    # match data
    matchMeanData= match[agg_col].transform('mean').replace([np.inf, np.NINF,np.nan], 0)
    all_data = pd.concat([all_data,matchMeanData.add_suffix('_matchMean')],axis=1)
    del matchMeanData
    gc.collect()

    # match max
    all_data = pd.merge(all_data, match[agg_col].agg('max').reset_index(), suffixes=["", "_matchMax"], how='left', on=['matchId'])

    # match std
    all_data = pd.merge(all_data, match[agg_col].agg('std').reset_index().replace([np.inf, np.NINF,np.nan], 0), suffixes=["", "_matchSTD"], how='left', on=['matchId'])

    all_data['matchType'] = all_data['matchType'].map({
    'crashfpp':"duo",
    'crashtpp':"duo",
    'duo':"duo",
    'duo-fpp':"duo",
    'flarefpp':"squad",
    'flaretpp':"squad",
    'normal-duo':"duo",
    'normal-duo-fpp':"duo",
    'normal-solo':"solo",
    'normal-solo-fpp':"solo",
    'normal-squad':"squad",
    'normal-squad-fpp':"squad",
    'solo':"solo",
    'solo-fpp':"solo",
    'squad':"squad",
    'squad-fpp':"squad"
    })

    gc.collect()

    all_data['matchType'] = LabelEncoder().fit_transform(all_data['matchType'])
    all_data.drop(['Id','groupId','matchId'],axis = 1,inplace = True)

    return all_data








print("feature engineering...")
X_train = featureModify() 

print('spliting data...')
X_train, X_test, y_train, y_test = train_test_split(X_train.drop(['winPlacePerc'],axis = 1), X_train['winPlacePerc'], test_size=0.1, random_state=42)




model = lgb.LGBMRegressor()

adj_params = {'n_estimators': range(100, 400, 10),
              'max_depth': range(5, 15, 2),
              'num_leaves': range(10,100,10),
              'min_child_weight': range(3, 20, 2),
              'min_child_samples': range(10, 30),
              'reg_lambda': [0,0.001,0.01,0.05,0.08,0.1,0.3,0.5],
              'reg_alpha': [0,0.001,0.01,0.05,0.08,0.1,0.3,0.5],
              'feature_fraction':[0.5,0.6,0.7,0.8,0.9],
              'bagging_fraction':[0.6,0.7,0.8,0.9,1.0]
             }

print('random search cv...')
rscv_start = time.time()
rscv = RandomizedSearchCV(model,adj_params, scoring='neg_mean_squared_error', cv=5)
rscv.fit(X_train, y_train)
rscv_end = time.time()
with open('rscv.txt','w') as file:
    print("best_params_: ", rscv.best_params_, file=file)
    print("best_score_: ", rscv.best_score_, file=file)

model = rscv.best_estimator_
model.learning_rate = 0.01

print('training...')
train_start = time.time()
model.fit(X_train,y_train)
train_end = time.time()

joblib.dump(model,'PUBG_lgbm.txt')

with open('time.txt','w') as file:
    print("Random SearchCV: ", rscv_end - rscv_start, file=file)
    print('trainning time: ', train_end - train_start ,file = file)

print('predicting...')
y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
mse = mean_squared_error(y_test, y_pred)
with open('result.txt','w') as file:
    print(y_pred, file = file)
    print("mse: ", mse,file = file)




