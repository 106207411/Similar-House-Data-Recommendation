Find similar houses
============================

### 資料夾架構
    .
    ├── find_similar_1model.py	      # 使用1model查詢相似案例的API
    ├── find_similar_3model.py        # 使用3model查詢相似案例的API
    ├── find_similar_batch.py         # 批次查詢相似案例=>匯出csv
    ├── train.py                      # 模型建立主程式
    ├── preprocess.py                 # 前處理工具
    ├── validation.py                 # 驗證工具
    ├── kfold_metrics.csv             # Kfold驗證指標分數
    ├── recommendation_list.csv       # 批次處理相似案例清單
    ├── trade_data_newtaipei.csv      # 實價登錄資料
    ├── reco_data_newtaipei.csv       # 建議案例紀錄
    └── model_info.sav                # 模型資訊

### 操作說明
#### train.py
```console
python train.py <path/to/trade_data> -n 1 -p 0.1
```

輸入必要參數: trade_data_newtaipei.csv

輸入選填參數: -n 1(只建立一個模型)、-n 3(依建物型態如公寓大樓套房建立三個模型)、-p 0.1(價格後處理10%)。預設參數即為-n 1 -p 0.1

輸出: model_info.sav、kfold_metrics.csv

#### find_similar_batch.py
```console
python find_similar_batch.py <path/to/appraisal_file> -m model_info.sav
```
輸入必要參數: reco_data_newtaipei.csv

輸入選填參數: -m model_info.sav (用一個模型推薦)、-m 3model_info.sav(用三個模型推薦)。預設參數即為-m model_info.sav

輸出: recommendation_list.csv

#### find_similar_1model.py
使用者輸入API，鑑價資訊須包含8種
town
build_area_adj
park_area
house_year
trans_floor
total_floor
x
y
#### find_similar_3model.py
使用者輸入API，鑑價資訊須包含9種
town
build_type
build_area_adj
park_area
house_year
trans_floor
total_floor
x
y
