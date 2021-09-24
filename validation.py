import preprocess
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from train import knn_model_fit


def get_test_case(trade_data, seq_no):
    """post-processing"""
    # Keep the order same as top n results
    test_case = trade_data.iloc[pd.Index(trade_data['seq_no']).get_indexer(seq_no)]
    test_case = test_case[["build_u_price_adj",
                           "adj_build_type",
                           "town",
                           "build_area_adj",
                           "park_area",
                           "house_year",
                           "trans_floor",
                           "total_floor",
                           "x",
                           "y"]]
      
    return test_case

def get_mape_kfold(test_case, new_reco6):
    """calculate the MAPE"""
    test_price = test_case.build_u_price_adj.values
    new_price = new_reco6.build_u_price_adj
    mape = sum(abs(new_price - test_price) / test_price) / len(new_price)
    return mape

def get_val_count_kfold(test_case, new_reco6, rate = 0.05):
    """val = 1 if test price falls in ±5% range of 6 new price average """
    test_price = test_case.build_u_price_adj.values
    new_price = new_reco6.build_u_price_adj
    mean_new_price = np.average(new_price)
    val = 1 if abs(mean_new_price - test_price) <= round(mean_new_price*rate, 1) else 0
    
    return val

def get_hit_rate_kfold(test_case, new_reco6, rate = 0.1):
    """calculate the hitrate"""
    test_price = test_case.build_u_price_adj.values
    new_price = new_reco6.build_u_price_adj
    hit_count= sum(abs(new_price - test_price).values <= test_price*rate) / len(new_price)
    
    return hit_count 

def get_new_reco6(trade_data_type, seq_no, price_std_dict, price_filter = 0):
    """post-processing"""
    # Keep the order same as top n results
    top100 = trade_data_type.iloc[pd.Index(trade_data_type['seq_no']).get_indexer(seq_no)]
    level_price = top100.iloc[0].build_u_price_adj

    # Filter ± town std of the first reco house price from top-100 reco data
    if price_filter == "town_std":
        town = top100.iloc[0].town
        if town not in price_std_dict.keys():
            std = 5
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
    new_reco6 = new_topn[["build_u_price_adj",
                          "seq_no",
                          "town",
                          "build_area_adj",
                          "park_area",
                          "house_year",
                          "trans_floor",
                          "total_floor",
                          "x",
                          "y"]][:6]
    

    return new_reco6


def get_kfold_metrics(trade_data, n_model, price_filter):
    """run kfold and return metric scores dict"""
    mape_all = 0
    mape_1 = 0
    mape_2 = 0
    mape_3 = 0
    val_score_all = 0
    val_score_1 = 0
    val_score_2 = 0
    val_score_3 = 0
    hit_rate10_all = 0
    hit_rate10_1 = 0
    hit_rate10_2 = 0
    hit_rate10_3 = 0 
    hit_rate20_all = 0
    hit_rate20_1 = 0
    hit_rate20_2 = 0
    hit_rate20_3 = 0 

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    if n_model == 3:
        trade_data_type1, trade_data_type2, trade_data_type3 = preprocess.preprocess_trade_data(trade_data, n_model)
        
        # iterate 3 types
        for idx, trade_data_type in tqdm(enumerate([trade_data_type1, trade_data_type2, trade_data_type3]), total = len([trade_data_type1, trade_data_type2, trade_data_type3]), desc='total'):
            price_std_dict = preprocess.get_price_std(trade_data_type)
            scaler = MinMaxScaler()
            X_train_bulid_type, y_train_bulid_type = preprocess.X_y_split(trade_data_type)
            X_train_bulid_type = scaler.fit_transform(X_train_bulid_type)

            mape = 0
            val_count = 0
            hit_rate10 = 0
            hit_rate20 = 0

            # iterate 10 times
            for train_index, test_index in tqdm(kf.split(X_train_bulid_type), total = 10, desc="kfold"):
                X_train, X_test = X_train_bulid_type[train_index], X_train_bulid_type[test_index]
                y_train, y_test = y_train_bulid_type[train_index], y_train_bulid_type[test_index]
                knn_model = knn_model_fit(X_train, y_train, n=6)
                # predict for each case
                for i in range(len(X_test)):
                    neighbors_id = knn_model.kneighbors([X_test[i]], 100, False)[0]
                    seq_no = [y_train[idx] for idx in neighbors_id]
                    test_case = get_test_case(trade_data, [y_test[i]])
                    new_reco6 = get_new_reco6(trade_data, seq_no, price_std_dict, price_filter)
                    mape += get_mape_kfold(test_case, new_reco6)
                    hit_rate10 += get_hit_rate_kfold(test_case, new_reco6, rate=0.1)
                    hit_rate20 += get_hit_rate_kfold(test_case, new_reco6, rate=0.2)
                    val_count += get_val_count_kfold(test_case, new_reco6)
                
            if idx+1 == 1:
                mape_1 = mape / len(trade_data_type)
                val_score_1 = val_count / len(trade_data_type)
                hit_rate10_1 = hit_rate10 / len(trade_data_type)
                hit_rate20_1 = hit_rate20 / len(trade_data_type)
            if idx+1 == 2:
                mape_2 = mape / len(trade_data_type)
                val_score_2 = val_count / len(trade_data_type)
                hit_rate10_2 = hit_rate10 / len(trade_data_type)
                hit_rate20_2 = hit_rate20 / len(trade_data_type)
            if idx+1 == 3:
                mape_3 = mape / len(trade_data_type)
                val_score_3 = val_count / len(trade_data_type)
                hit_rate10_3 = hit_rate10 / len(trade_data_type)
                hit_rate20_3 = hit_rate20 / len(trade_data_type)
                
            mape_all += mape
            val_score_all += val_count
            hit_rate10_all += hit_rate10
            hit_rate20_all += hit_rate20
            
        mape_all /= (len(trade_data_type1) + len(trade_data_type2) + len(trade_data_type3))
        val_score_all /= (len(trade_data_type1) + len(trade_data_type2) + len(trade_data_type3))
        hit_rate10_all /= (len(trade_data_type1) + len(trade_data_type2) + len(trade_data_type3))
        hit_rate20_all /= (len(trade_data_type1) + len(trade_data_type2) + len(trade_data_type3))
        
    if n_model == 1:
        trade_data_type1, trade_data_type2, trade_data_type3 = preprocess.preprocess_trade_data(trade_data, n_model = 3)
        trade_data_type = preprocess.preprocess_trade_data(trade_data, n_model)
        price_std_dict = preprocess.get_price_std(trade_data_type)
        scaler = MinMaxScaler()
        X_train_bulid_type, y_train_bulid_type = preprocess.X_y_split(trade_data_type)
        X_train_bulid_type = scaler.fit_transform(X_train_bulid_type)


        # iterate 10 times
        for train_index, test_index in tqdm(kf.split(X_train_bulid_type), total = 10, desc="kfold"):
            X_train, X_test = X_train_bulid_type[train_index], X_train_bulid_type[test_index]
            y_train, y_test = y_train_bulid_type[train_index], y_train_bulid_type[test_index]
            knn_model = knn_model_fit(X_train, y_train, n=6)

            # predict for each case
            for i in range(len(X_test)):
                neighbors_id = knn_model.kneighbors([X_test[i]], 100, False)[0]
                seq_no = [y_train[idx] for idx in neighbors_id]
                test_case = get_test_case(trade_data, [y_test[i]])
                new_reco6 = get_new_reco6(trade_data, seq_no, price_std_dict, price_filter)
                mape = get_mape_kfold(test_case, new_reco6)
                hit_rate10 = get_hit_rate_kfold(test_case, new_reco6, rate=0.1)
                hit_rate20 = get_hit_rate_kfold(test_case, new_reco6, rate=0.2)
                val_count = get_val_count_kfold(test_case, new_reco6)

                if test_case.adj_build_type.values == 1:
                    mape_1 += mape 
                    val_score_1 += val_count 
                    hit_rate10_1 += hit_rate10 
                    hit_rate20_1 += hit_rate20 
                if test_case.adj_build_type.values == 2:
                    mape_2 += mape 
                    val_score_2 += val_count 
                    hit_rate10_2 += hit_rate10 
                    hit_rate20_2 += hit_rate20 
                if test_case.adj_build_type.values == 3:
                    mape_3 += mape 
                    val_score_3 += val_count 
                    hit_rate10_3 += hit_rate10 
                    hit_rate20_3 += hit_rate20 
    
                mape_all += mape
                val_score_all += val_count
                hit_rate10_all += hit_rate10
                hit_rate20_all += hit_rate20

        mape_1 /= len(trade_data_type1)
        mape_2 /= len(trade_data_type2)
        mape_3 /= len(trade_data_type3)
        val_score_1 /= len(trade_data_type1)
        val_score_2 /= len(trade_data_type2)
        val_score_3 /= len(trade_data_type3)
        hit_rate10_1 /= len(trade_data_type1)
        hit_rate10_2 /= len(trade_data_type2)
        hit_rate10_3 /= len(trade_data_type3)
        hit_rate20_1 /= len(trade_data_type1)
        hit_rate20_2 /= len(trade_data_type2)
        hit_rate20_3 /= len(trade_data_type3)
        mape_all /= len(trade_data_type)
        val_score_all /= len(trade_data_type)
        hit_rate10_all /= len(trade_data_type)
        hit_rate20_all /= len(trade_data_type)
        
    score_dict = {"val_score_all": val_score_all, 
                "val_score_1": val_score_1, 
                "val_score_2": val_score_2, 
                "val_score_3": val_score_3,
                "mape_all": mape_all, 
                "mape_1": mape_1, 
                "mape_2": mape_2, 
                "mape_3": mape_3, 
                "hit_rate10_all": hit_rate10_all, 
                "hit_rate10_1": hit_rate10_1, 
                "hit_rate10_2": hit_rate10_2, 
                "hit_rate10_3": hit_rate10_3,
                "hit_rate20_all": hit_rate20_all, 
                "hit_rate20_1": hit_rate20_1, 
                "hit_rate20_2": hit_rate20_2, 
                "hit_rate20_3": hit_rate20_3}
        
    return score_dict       
