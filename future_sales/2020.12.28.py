import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from xgboost import plot_importance


test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
# items: item_name, item_id, item_category_id
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv") 
# categories: item_category_name, item_category_id
categories = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
# train: date, date_block_num, shop_id, item_id, item_price, item_cnt_day**
train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
# shops: shop_name, shop_id
shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")


def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


