import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
import seaborn as sns
from sklearn import preprocessing
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')
df.drop(["Unnamed: 0","team_id","team_name"], axis=1, inplace=True)
df.dropna(subset=["shot_id_number"],inplace=True)


obj_li = ["area_of_shot",
          "shot_basics",
          "range_of_shot",
          "home/away",
          "game_season",
          "type_of_shot",
          "type_of_combined_shot"]
num_li = ["remaining_min.1",
          "power_of_shot.1",
          "knockout_match.1",
          "remaining_sec.1",
          "distance_of_shot.1",
          "location_x",
          "location_y",
          "remaining_min",
          "power_of_shot",
          "knockout_match",
          "remaining_sec",
          "distance_of_shot"]
df["lat/lng"].fillna("0, 0", inplace=True)
for i in obj_li:
    df[i].fillna("unknown",inplace=True)
    le = preprocessing.LabelEncoder()
    df[i] = le.fit_transform(df[i])

for i in num_li:
    df[i].fillna(-1,inplace=True)

df["date_of_game"].fillna(df["date_of_game"].iloc[0],inplace=True)
df["date_of_game"] = pd.to_datetime(df["date_of_game"])
df["year"] = df["date_of_game"].apply(lambda x:x.year)
df["day"] = df["date_of_game"].apply(lambda x:x.day)
df["month"] = df["date_of_game"].apply(lambda x:x.month)

def segregate(x):
    l = x["lat/lng"].split(", ")
    return pd.Series(l)
df[["lat","long"]] = df.apply(segregate, axis=1)
df["lat"] = pd.to_numeric(df["lat"])
df["long"] = pd.to_numeric(df["long"])
df["addition_cords"] = df["long"] + df["lat"]
df["sub_cords"] = df["long"] - df["lat"]
df["add_loc"] = df["location_x"] + df["location_y"]
df["sub_loc"] = df["location_x"] - df["location_y"]


train_df = df[df["is_goal"].notnull()].copy()
test_df = df[df["is_goal"].isnull()].copy()


features = [c for c in train_df.columns if c not in ['match_id', "match_event_id", 'is_goal',"shot_id_number", "date_of_game", "lat/lng"]]
target = train_df['is_goal']
param = {
    'bagging_freq': 5,          'bagging_fraction': 0.335,   'boost_from_average':'false',   'boost': 'random_forest',
    'feature_fraction': 0.41,   'learning_rate': 0.1,     'max_depth': -1,                'metric':'auc',
    'min_data_in_leaf': 60,     'min_sum_hessian_in_leaf': 10.0,'num_leaves': 15,           'num_threads': 8,
    'tree_learner': 'serial',   'objective': 'binary',      'verbosity': 1
}
folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=2319)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])
    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

sub = pd.DataFrame({"shot_id_number": test_df.shot_id_number})
sub["is_goal"] = predictions
sub.to_csv("submission.csv", index=False)