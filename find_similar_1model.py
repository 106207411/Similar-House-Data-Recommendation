from datetime import timedelta
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib

app = FastAPI()
model_info = joblib.load("./model_info.sav")

class Appraisal(BaseModel):
    town: str
    build_area_adj: float
    park_area: float
    house_year: float
    trans_floor: int
    total_floor: int
    x: float
    y: float

@app.post("/find-similar-1model/")
def get_top6_case(appraisal:Appraisal):
    """find 6 similar house"""

    # used for post-preprocessing
    kneighbors = 100
    town = appraisal.town
    X_test = [[appraisal.build_area_adj, appraisal.park_area, appraisal.house_year, appraisal.trans_floor, appraisal.total_floor, appraisal.x, appraisal.y]]
    
        
    # read model info
    knn_model = model_info["model"]
    scaler = model_info["scaler"]
    y_train = model_info["seq"]
    price_std_dict = model_info["price_std"]

    # Apply minmaxScaler
    X_test_norm = scaler.transform(X_test)
    # Get top-100 recommeded house id
    neighbors_id = knn_model.kneighbors(X_test_norm, kneighbors, False)

    # mapping neighbors_id to 1D array
    # get top100 seq_no
    seq_no = y_train[neighbors_id[0]] 

    """post-processing to get top6 case from raw dataframe"""

    # Keep the order same as top 100 results
    trade_data = model_info["trade_data"]
    top100 = trade_data.iloc[pd.Index(trade_data['seq_no']).get_indexer(seq_no)]

    level_price = top100.iloc[0].build_u_price_adj
    price_filter = model_info["price_filter"]

    # Filter ± town std of the first reco house price from top-100 reco data
    if price_filter == "town_std":
        town = top100.iloc[0].town
        
        # if appraisal town isn't seen in the training data
        # then we give it 5w as town std
        if town not in price_std_dict.keys():
            std = 5

        # if town std is nan
        # then we give it 5w as town std
        else:
            std = 5 if np.isnan(price_std_dict[town]) else price_std_dict[town]
        new_topn = top100[(top100.build_u_price_adj <= level_price + std) & (top100.build_u_price_adj >= level_price - std)]

    elif price_filter == "none":
        new_topn = top100

    else:
        price_filter = float(price_filter)
        # Filter (ex. ±7.5w) of the first reco house price from top-100 reco data
        if price_filter >= 1:
            new_topn = top100[(top100.build_u_price_adj <= level_price + price_filter) & (top100.build_u_price_adj >= level_price - price_filter)]
        
        # Filter (ex. ±10%) of the first reco house price from top-100 reco data
        if price_filter > 0 and price_filter < 1:
            new_topn = top100[(top100.build_u_price_adj <= level_price * (1 + price_filter)) & (top100.build_u_price_adj >= level_price * (1 - price_filter))]

    
    # Select top-6 houses as final reco
    new_reco6 = new_topn[["seq_no",
                          "build_u_price_adj",
                         "town",
                         "build_type",
                         "build_area_adj",
                         "park_area",
                         "house_year",
                         "trans_floor",
                         "total_floor",
                         "x",
                         "y"]][:6]

    return {"detail":new_reco6.to_dict("records")}