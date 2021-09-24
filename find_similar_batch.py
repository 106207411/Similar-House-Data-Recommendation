import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import argparse
import os
import joblib


def dump_reco_list(appraisal_data, model_info): 
    """Dumping final list of 6 recommendation houses for each appraisal case
       Return recommendation_list.csv"""
    output_path = 'recommendation_list.csv'
    
    for idx in tqdm(range(len(appraisal_data)), desc="dump to list"):
        # select appraisal data one by one
        # get top-6 reco data seq_no
        appraisal = appraisal_data.iloc[[idx]]
        top6_case = get_top6_case(model_info, appraisal)
        appraisal = appraisal[["CASE_ID", "PRICE", "TOWN", "BUILD_TYPE", "BUILD_AREA_ADJ", "PARK_AREA", "HOUSE_YEAR", "TRANS_FLOOR", 
                            "TOTAL_FLOOR", "LONGITUDE", "LATITUDE"]]
        appraisal.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
        top6_case.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)


def get_top6_case(model_info, appraisal, kneighbors = 100):
    """get 6 recommeded house 
       if n_model is 3, 6 similar cases are selected based on 3 build types"""

    # used for post-preprocessing
    build_type = appraisal["BUILD_TYPE"].values[0]
    town = appraisal["TOWN"].values[0]
    
    # convert appraisal data build_type to trade data build_type
    # apartment
    if build_type in ["1", "H"]:
        build_type = 1

    # complex building
    if build_type in ["3", "5", "F"]:
        build_type = 2

    # suite
    if build_type in ['G']:
        build_type = 3

    X_test = appraisal[["BUILD_AREA_ADJ", "PARK_AREA", "HOUSE_YEAR", "TRANS_FLOOR", "TOTAL_FLOOR", "LONGITUDE", "LATITUDE"]]

    if model_info["n_model"] == 3:
        knn_model = model_info["model_dict"][f"build_type{build_type}"]
        scaler = model_info["scaler_dict"][f"build_type{build_type}"]
        y_train = model_info["seq_dict"][f"build_type{build_type}"]
        price_std_dict = model_info["price_std_dict"][f"build_type{build_type}"]
        
    if model_info["n_model"] == 1:
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

    # Keep the order same as top n results
    if model_info["n_model"] == 3:
        trade_data = model_info["trade_data_dict"][f"build_type{build_type}"]
        top100 = trade_data.iloc[pd.Index(trade_data['seq_no']).get_indexer(seq_no)]
    if model_info["n_model"] == 1:
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

    return new_reco6

def read_appraisal(appraisal_file_name):
    appraisal_data = pd.read_csv(appraisal_file_name)
    appraisal_data = appraisal_data[["PRICE", "CASE_ID", "TOWN", "BUILD_TYPE", "BUILD_AREA_ADJ", "PARK_AREA", "HOUSE_YEAR", "TRANS_FLOOR", "TOTAL_FLOOR", "LONGITUDE", "LATITUDE"]]
    appraisal_data = appraisal_data.dropna()
    
    return appraisal_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("appraisal_file_name", help="input appraisal data (.csv)")
    parser.add_argument("-m", "--model_info", help="input model info (.sav)", choices=["model_info.sav", "3model_info.sav"], default="model_info.sav")
    arg = parser.parse_args()
    model_info = joblib.load(arg.model_info)

    appraisal_data = read_appraisal(arg.appraisal_file_name)
    dump_reco_list(appraisal_data, model_info)