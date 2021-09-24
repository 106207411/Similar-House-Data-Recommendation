import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def boolean_based_on_note(df, a1s, a0s, a11s):
    boolean_selected = pd.Series(np.array([False]*df.shape[0]))
    for a1 in a1s:        
        boolean_selected = (boolean_selected | df["remark"].str.contains(a1, na=False))  #包含此文字內容會進入備註欄==true
        
    for a0 in a0s:
        boolean_selected = (boolean_selected & ~df["remark"].str.contains(a0, na=False)) #包含此文字不會進入備註欄==false
        
    for a11 in a11s:
        boolean_selected = (boolean_selected | df["remark"].str.contains(a11, na=False))   #包含此文字內容會進入備註欄==true
        
    return boolean_selected

def set_note_flag(df, flag_name, a1s, a0s, a11s):
    df[flag_name] = False
    df.loc[boolean_based_on_note(df, a1s, a0s, a11s), flag_name] = True

def exclude_special_case(df):
    # filter target case
    df = df[( (df['target_type']=='房地(土地+建物)') | (df['target_type']=='房地(土地+建物)+車位') )]
    df = df[( (df['trans_floor'] > 0) & (df['build_u_price_adj'] > 0) )]

    set_note_flag(df, "flag_毛胚屋", 
                ['無.*隔間','毛胚','房間需自行隔間','未隔間'], 
                [], 
                [])

    set_note_flag(df, "flag_瑕疵物件", 
                ['瑕疵物件','海砂屋','受損房屋','老舊','凶宅','祭祀公業','事故','非自然','無接水電','損壞','破損','滲*水','壁癌','屋況不佳','屋況.*差','屋況嚴重瑕疵','房屋.*半毀','房屋破舊','屋況需整理','屋況殘破','屋況不良','房屋.*已坍塌','房屋老舊','建物須整修','部分土地有人死亡','入口處狹小巷弄狹小','無接水接電','高壓電線正下方','多年無人使用','建物已毀損','死亡','事故','年久失修','廢墟','無法居住','面積流失','荒廢','河床侵蝕'], 
                [], 
                [])

    set_note_flag(df, "flag_親友、員工或其他特殊關係", 
                ['親友','朋友','親屬','親戚','近親','員工','特殊關係','關係戶','親等','等親','朋友關係','兄','弟','姐','姊','妹','叔','姪','母','父','女','子','祖','孫','女兒','直系血親','股東','好友','姑','關係人','關係企業','伯','媳','同事','夫','妻','嫂','監察人','董事','婆媳','鄰居間交易','鄰居關係','信徒與寺廟','信徒與宮廟','配偶','姻親','熟人','房東.*房客','承租戶購買','承租戶承買','公益交易','特惠戶','買方為地上房屋所有權人','合夥關係','地主戶'], 
                ['車位','子母車位','股東股權讓渡','九份子'], 
                [])

    set_note_flag(df, "flag_急買急賣", 
                ['急買','急賣','賣方急錢用','急售','急換屋需資金','急著出售','急出售'], 
                [], 
                [])
    
    # 根據flag欄位排除特殊交易資料
    # flag欄位中必須都是false
    df = df[~df.filter(regex="flag_.*").any(axis="columns")]
      
    # 針對含增建或未登記建物時，只需排除頂樓或一樓案例
    # 因為大多中間層都會有這種情況(如:陽台外推)
    set_note_flag(df, "flag_含增建或未登記建物", 
                ['增建','未.*登記','未.*保存','無保存','未辦保登','無登記','加蓋','外推','加建','外移','頂加','鐵皮屋','鐵皮厝','增建.*農業資材室','增建.*儲藏室'], 
                ['不包括.*增建','不包括.*未登記','不含.*未登記','未包含.*未保存','無增建','未分割','未.*繼承','補辦登記','未辦理過戶登記','未辦移轉','祖產未登記','未移轉登記','未償債務','不包含未保存','不含未登記','分次登記','未包含在本次土地買賣價金內','地上.*鐵皮屋','含鐵皮屋','未保存建物占用','鄰地.*增建','車位.*面積未登記','尚未成立管委會','抵償','建物.*不計價','不計入價金','未計算價金','未計入土地交易總價中','未辦簽約僅辦理登記','未辦登記僅辦簽約','僅.*產權移轉登記','代理辦理登記','地政士.*並未代理撰擬不動產買賣契約書','無登記在書狀內','不另計價','增建不計價','附贈.*增建','贈與未登記建物','加蓋為0','未辦之借名登記','未登記建物所有權人購買土地','地上未登記建物為買方所有','稅金問題遲未登記','僅受託買賣案件申請登記','增建部分占用國產署土地','未拆分單價'], 
                ['未辦保存登記及未辦繼承登記','繼承後','未保存建物'])

    boolean_top_and_first_floor = ((df.trans_floor == 1) | (df.trans_floor == df.total_floor))
    boolean_extra_build = (df["flag_含增建或未登記建物"] == True)
    df_no_specialcase = df[~(boolean_extra_build & boolean_top_and_first_floor)]
    
    return df_no_specialcase    

def filter_by_adj_build_type(df):
    df1 = df[df.adj_build_type == 1]
    df2 = df[df.adj_build_type == 2]
    df3 = df[df.adj_build_type == 3]
    
    # filter by PATTERN_ROOM and PATTERN_BATH 
    # df1 = df1[(df1.pattern_room < 4) & (df1.pattern_bath < 4)]
    # df2 = df2[(df2.pattern_room < 5) & (df2.pattern_bath < 5)]
    
    return df1, df2, df3

def select_var(df):
    df = df[["seq_no",
             "build_u_price_adj",
             "town",
             "build_type",
             "build_area_adj",
             "park_area",
             "house_year",
             "trans_floor",
             "total_floor",
             "x",
             "y"]]
    return df

def preprocess_trade_data(trade_data, n_model):
    '''Preprocess trade data
       Split into 3 data based on build type if n_model is 3'''
    trade_data_no_specialcase = exclude_special_case(trade_data)
    if n_model == 3:
        trade_data_type1, trade_data_type2, trade_data_type3 = filter_by_adj_build_type(trade_data_no_specialcase)
        trade_data_type1 = select_var(trade_data_type1).dropna()
        trade_data_type2 = select_var(trade_data_type2).dropna()
        trade_data_type3 = select_var(trade_data_type3).dropna()
        return trade_data_type1, trade_data_type2, trade_data_type3
    if n_model == 1:
        trade_data = select_var(trade_data_no_specialcase).dropna()
        return trade_data

def get_price_std(trade_data_type):
    price_std_dict = trade_data_type.groupby("town")["build_u_price_adj"].agg("std").to_dict()
    return price_std_dict

def X_y_split(df):
    X = df[["build_area_adj",
            "park_area",
            "house_year",
            "trans_floor",
            "total_floor",
            "x",
            "y"]]
    y = df[["seq_no"]]
    y = y.values.flatten()
    
    return X, y