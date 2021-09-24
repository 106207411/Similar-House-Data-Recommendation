import preprocess
import validation
import csv
import pandas as pd
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib


def knn_model_fit(X_train, y_train, n=6):
    """Using L2-dist to measure similarity"""
    clf_knn = KNeighborsClassifier(n_neighbors=n, algorithm="ball_tree", weights="distance")
    clf_knn.fit(X_train, y_train)

    return clf_knn

def model_train_final(trade_data_raw, n_model, price_filter):
    """train model 
       output: 
       1. model info (.sav) 
       2. kfold metrics (.csv)"""

    if n_model == 3:
        """train model"""
        trade_data_type1, trade_data_type2, trade_data_type3 = preprocess.preprocess_trade_data(trade_data_raw, n_model)

        # get price std for each town 
        price_std_build_type1 = preprocess.get_price_std(trade_data_type1)
        price_std_build_type2 = preprocess.get_price_std(trade_data_type2)
        price_std_build_type3 = preprocess.get_price_std(trade_data_type3)
        
        # split training data into 3 types and normalize X_train
        # preserve scaler info to apply on X_test in the future
        X_train_build_type1, y_train_build_type1 = preprocess.X_y_split(trade_data_type1)
        X_train_build_type2, y_train_build_type2 = preprocess.X_y_split(trade_data_type2)
        X_train_build_type3, y_train_build_type3 = preprocess.X_y_split(trade_data_type3)
        scaler_build_type1 = MinMaxScaler()
        scaler_build_type2 = MinMaxScaler()
        scaler_build_type3 = MinMaxScaler()
        X_train_build_type1 = scaler_build_type1.fit_transform(X_train_build_type1)
        X_train_build_type2 = scaler_build_type2.fit_transform(X_train_build_type2)
        X_train_build_type3 = scaler_build_type3.fit_transform(X_train_build_type3)

        # train 3 knnn models and memorize training data 
        knn_build_type1 = knn_model_fit(X_train_build_type1, y_train_build_type1, n=6)
        knn_build_type2 = knn_model_fit(X_train_build_type2, y_train_build_type2, n=6)
        knn_build_type3 = knn_model_fit(X_train_build_type3, y_train_build_type3, n=6)

        # dump all the training history
        model_info_dict = {"model_dict": {"build_type1": knn_build_type1, 
                                             "build_type2": knn_build_type2,
                                             "build_type3": knn_build_type3},
                              "scaler_dict": {"build_type1": scaler_build_type1,
                                              "build_type2": scaler_build_type2,
                                              "build_type3": scaler_build_type3},
                              "seq_dict": {"build_type1": y_train_build_type1,
                                           "build_type2": y_train_build_type2,
                                           "build_type3": y_train_build_type3},
                              "price_std_dict": {"build_type1": price_std_build_type1,
                                                 "build_type2": price_std_build_type2,
                                                 "build_type3": price_std_build_type3},
                              "trade_data_dict": {"build_type1": trade_data_type1,
                                                  "build_type2": trade_data_type2,
                                                  "build_type3": trade_data_type3},
                              "n_model": n_model,
                              "price_filter": price_filter}
        
        # save model info to file
        joblib.dump(model_info_dict, '3model_info.sav')

    if n_model == 1:
        trade_data = preprocess.preprocess_trade_data(trade_data_raw, n_model)
                
        # get price std for each town 
        price_std = preprocess.get_price_std(trade_data)

        # normalize X_train
        # preserve scaler info to apply on X_test in the future
        X_train, y_train = preprocess.X_y_split(trade_data)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)

        # train models and memorize training data
        knn = knn_model_fit(X_train, y_train, n=6)
        
        # dump all the training history
        model_info_dict = {"model": knn,
                              "scaler": scaler,
                              "seq": y_train,
                              "price_std": price_std,
                              "trade_data": trade_data,
                              "n_model": n_model,
                              "price_filter": price_filter}
        
        # save model info to file
        joblib.dump(model_info_dict, 'model_info.sav')

            
    # calculate the metrics from kfold validation and save as csv
    metrics_dict = validation.get_kfold_metrics(trade_data_raw, n_model, price_filter)
    with open("kfold_metrics.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['current_metric_all',
                         'current_metric_apartment', 
                         'current_metric_complex',
                         'current_metric_suite',
                         'mape_all',
                         'mape_apartment',
                         'mape_complex',
                         'mape_suite',
                         'hit_rate10_all'
                         'hit_rate10_apartment',
                         'hit_rate10_complex',
                         'hit_rate10_suite',
                         'hit_rate20_all',
                         'hit_rate20_apartment',
                         'hit_rate20_complex',
                         'hit_rate20_suite'])
        writer.writerow(list(metrics_dict.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trade_file_name", help="input training data (.csv)")
    parser.add_argument("-n", "--n_model", help="specify the number of model to train", type=int, choices=[1, 3], default=1)
    parser.add_argument("-p", "--price_filter", help="specify filter-type for price post-preprocessing", choices=["0.1", "0.2", "2.5", "5", "7.5", "none", "town_std"], default="0.1") #remember to cast to float
    arg = parser.parse_args()

    trade_data_raw = pd.read_csv(arg.trade_file_name)
    model_train_final(trade_data_raw, arg.n_model ,arg.price_filter)

    
    