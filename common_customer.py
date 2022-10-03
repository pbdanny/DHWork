import sys
sys.path.insert(0, '/nfs/science/shared/ipythonNotebooks/thanakb/dh_python_fn/')
from dhSparkInit import *

from dhMail import sentMeMail
from dhLog import getLog
import dhSpark as ds

import pandas as pd
import numpy as np
import configparser
import os
import subprocess

import csv
import time
start_time = time.monotonic()

# -----------------------------
# Prod Preparation
# -----------------------------
# Expand promo prod
# into promo subclass
# ----------------------------

# Create promo prod list at subclass lv
def check_prod_list(prod_file):
    """
    From provided prod_code, auto converted into 9-digits with zero leading
    """
    df = pd.read_csv(prod_file, dtype=object)

    df_len = df['prod_code'].str.len()
    df_len_9digit = df_len == 9

    if df_len_9digit.all():
        print('all prod_code have 9 digits')
        df_out = df

    else:
        print('Convert prod_code to 9 digits')
        df_out= df['prod_code'].apply(lambda x : '{0:0>9}'.format(x))

    # save 9 digits
    df_out.to_csv(prod_file, header=True, index=False)

    return df_out

def get_promo_subcl(prod_file, result_prod_file, lv):
    """
    Load .csv of promo prod code, find promoted subclass from list, then
    create promo subclass from loaded list
    save as 'prod_list.csv'
    Parameter :
    prod_file : 'xxx.csv' of promo product with header 'prod_code',9-digits
                with leading zero
    lv        : level of analysis
    """
    # read prod list
    df = pd.read_csv(prod_file, dtype=object)
    promo_prod = ss.createDataFrame(df)

    # Check level of analysis
    if lv == 'product':
        print('Analysis on product level')

        # export return
        df.to_csv(result_prod_file, index = False)
        print('Copy product for analysis from {} to {}'.format(prod_file, result_prod_file))

    elif lv in ['subclass','class']:
        print('Analysis on "{}" level, convert prod list to prod list of "{}"'.format(lv.upper(), lv.upper()))

        if lv == 'subclass':
            qry_hier = 'prod_hier_l10_code'
            qry_hier_desc = 'prod_hier_l10_desc'
        elif lv == 'class':
            qry_hier = 'prod_hier_l20_code'
            qry_hier_desc = 'prod_hier_l20_desc'
        else:
            print('Unrecognized level of analysis')

        # get prod hier lv code
        promo_subcl = \
        (prod
        .join(broadcast(promo_prod), 'prod_code')
        .select(qry_hier)
        .distinct()
        )

        # Roll up to prod hier lv level
        promo_subcl_prod = \
        (prod
        .join(promo_subcl, qry_hier)
        .select(qry_hier, qry_hier_desc, 'prod_code')
        .distinct()
        )

        # Calculate percentate of promo sku vs rollup subcl prod set
        sku_promo_code = promo_prod.withColumnRenamed('prod_code', 'sku_prod_code')

        promo_subcl_prod_pv = \
        (promo_subcl_prod
        .join(broadcast(sku_promo_code), [promo_subcl_prod.prod_code == sku_promo_code.sku_prod_code], 'left')
        .withColumn('sku_promo_tag', F.when(F.col('sku_prod_code').isNotNull(), F.lit('promo_sku')).otherwise(F.lit('promo_subcl')))
        .groupBy(qry_hier, qry_hier_desc)
        .pivot('sku_promo_tag', ['promo_sku', 'promo_subcl'])
        .agg(F.count('prod_code'))
        )

        promo_subcl_prod_pv = promo_subcl_prod_pv.withColumn('promo_tt',  sum(promo_subcl_prod_pv[col] for col in ['promo_sku', 'promo_subcl']))
        promo_subcl_prod_pv = promo_subcl_prod_pv.withColumn('pct_promo_sku_tt', F.round((F.col('promo_sku')/F.col('promo_tt')*100), 2))
        n_rows = promo_subcl_prod_pv.count()

        print('Total promoted {} : {}'.format(lv, n_rows))
        promo_subcl_prod_pv.orderBy('pct_promo_sku_tt', ascending = False).show(n_rows, truncate=False)

        # List subclass , count promo products
        # promo_subcl_prod.groupBy('prod_hier_l10_desc').agg(F.count('prod_code').alias('n_prod')).show(truncate=False)

        # convert to pandas for export
        promo_subcl_prod = promo_subcl_prod.select('prod_code').distinct()
        subcl_df = ds.to_pandas(promo_subcl_prod)

        # export return & export
        subcl_df.to_csv(result_prod_file, index = False)
        print('Create analysis of product for analysis at {}'.format(result_prod_file))

    else:
        print('Unrecognized {} level of analysis'.format(lv))

def filter_day_of_week(txn, dow):
    """
    Map date_id with fis_day_of_week_num, and filter only day of week need
    """    
    dow_map = date.select(F.col('date_id').cast('date'), 'fis_day_of_week_num')
    
    if isinstance(txn, ds.itemFct):
        txn_df = txn.df
        out_df = (
            txn_df
            .join(broadcast(dow_map), 'date_id')
            .filter(F.col('fis_day_of_week_num').isin(dow))
        )
        txn.df = out_df
    
    else:
        out_df = (
            txn
            .join(broadcast(dow_map), 'date_id')
            .filter(F.col('fis_day_of_week_num').isin(dow))
        )
        return out_df

def filter_prod(txn, prod_code):
    """
    Filter only select prod
    """
    focus_prod = prod.filter(F.col('prod_code').isin(prod_code)).select('prod_code')
    if isinstance(txn, ds.itemFct):
        txn_df = txn.df
        out_df = txn_df.join(broadcast(focus_prod), 'prod_code')
        txn.df = out_df
    
    else:
        out_df = txn.join(broadcast(focus_prod), 'prod_code')
        return out_df

# -------
# Main
# -------

if __name__ == '__main__':

    # -----------------------
    # Path structure
    # -----------------------
    try:
        # define master path for interactive ipython
        __IPYTHON__
        master_path = os.path.dirname(os.getcwd())
    except NameError:
        # define master path for batching
        master_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    config_path = os.path.join(master_path, 'config')
    result_path = os.path.join(master_path, 'result')
    figure_path = os.path.join(master_path, 'result', 'figure')
    model_path = os.path.join(master_path, 'model')

    # ------------------------
    # Parameter for analysis
    # ------------------------
    config = configparser.ConfigParser(allow_no_value=True)

    config.read(os.path.join(config_path, 'EE_BDA.conf'))

    # project
    PROJ = config['project']['proj']

    # store format
    STR_FORMAT = config['store']['str_format']

    # num partition size
    # data 6 mth ~ 60MB fit 1 partition size
    # Usually, set 40 for pararallism (10 executor x 4 core)
    SAVE_PARTITION_SIZE = int(config['hive_config']['save_partition_size'])

    # Products
    PROD_SEED_FILE = config['products']['promo_seed_file']
    PROD_ANAL_FILE = config['products']['prod_list_file']

    # Analysis level
    ANALYSIS_LEVEL = config['products']['analysis_level']

    # -------------------------------
    # Check sales by day x prod x
    # -------------------------------
    
#     # Read encode list
#     encode = pd.read_csv(os.path.join(config_path, 'encode_prodlist_period_format_date.csv'), dtype={'prod_code':object})

#     # -----------
#     # Large format
#     # -----------
#     for row in encode.itertuples():
        
#         fmt = row.fmt.lower()
#         prd_str = row.prod_code
#         prd_gr_desc = row.prod_gr
#         prd_lst = list(prd_str.split(','))
#         dow = row.fis_day_of_week_num
        
#         periods = [{'yr_tag':'ty', 'str_wk':201910, 'end_wk':201951},
#                    {'yr_tag':'ly', 'str_wk':201810, 'end_wk':201851}]
        
#         for period in periods:
            
#             year = period['yr_tag'].upper()
#             str_wk = period['str_wk']
#             end_wk = period['end_wk']
#             txn = ds.itemFct('fis_week_id', str_wk, end_wk, fmt)
            
#             # mapping and filter
#             filter_prod(txn, prd_lst)
#             filter_day_of_week(txn, dow)
        
#             # get result
#             out = (
#                 txn
#                 .df
#                 .withColumn('year', F.lit(year))
#                 .withColumn('format', F.lit(fmt))
#                 .withColumn('prod_list', F.lit(prd_str))
#                 .withColumn('prod_gr', F.lit(prd_gr_desc))
#                 .groupBy('year', 'format', 'prod_list', 'prod_gr' 'date_id', 'fis_day_of_week_num')
#                 .agg(*kpi)
#             )
            
#             out_df = ds.to_pandas(out)
            
#             # write result
#             with open(os.path.join(result_path, 'check1.csv'), mode = 'a') as f:
#                 out_df.to_csv(f, mode='a', header=f.tell()==0, index = False)                
    
    # -------------------------
    # find commond customer
    # -------------------------
    encode = pd.read_csv(os.path.join(config_path, 'encode_prodlist_period_format_date.csv'), dtype={'prod_code':object})
    
    card_weeks = list()
    
    for row in encode.itertuples():
        
        fmt = row.fmt.lower()
        prd_str = row.prod_code
        prd_gr_desc = row.prod_gr
        prd_lst = list(prd_str.split(','))
        date_id = row.date_id
        
        txn = ds.itemFct('fis_week_id', 201950, 201951, fmt, cust_seg=False)
        filter_prod(txn, prd_lst)
        card_prod = (
            txn
            .df.filter(F.col('date_id').isin(date_id))
            .filter(F.col('card_id').isNotNull())
            .select('card_id')
            .drop_duplicates()
        )
        
        card_weeks.append({'prd_gr':prd_gr_desc, 'df':card_prod})
    
    # Start test combinaion
    
    from itertools import combinations
    
    N = len(card_weeks)
    
    with open(os.path.join(result_path,'comb_out.csv'), mode='w') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['comb_flag', 'n_custs'])
    
        for r in range(1, N+1):

            combs = combinations(card_weeks, r)

            for comb in combs:

                combs_flag = str()

                for i, gr_df in enumerate(comb):

                    if i == 0:
                        combs_flag = gr_df['prd_gr']
                        chain_cc = gr_df['df']

                    else:
                        combs_flag = combs_flag + ' , ' + gr_df['prd_gr']
                        chain_cc = chain_cc.join(gr_df['df'], 'card_id')

                n_cc = chain_cc.count()

                print('Combination "{}" : {}'.format(combs_flag, n_cc))

                writer.writerow([combs_flag, n_cc])
    
    ################################################
    # -----------
    # Finalizing
    # -----------

    print('===========================')
    print('Data check finish')
    print('===========================')

    # Calculate elapse time & sent mail
    elapsed_time = time.monotonic() - start_time
    str_elapse_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    script_out_file = os.path.realpath(__file__)+'.txt'
    sentMeMail('Step {} finish'.format(__file__),
            body='Elapse : ' + str_elapse_time,
            file=script_out_file)
    ss.stop()

    
#     # Check Tesco Water
#     prod.filter(F.col('prod_code').isin('012231304','012717851')).select('prod_code', 'prod_alt_desc').show(10, False)

#     # what is DM
#     prod.filter(F.col('prod_alt_desc').startswith('TESCO DRINKING WATER 1500')).select('prod_code', 'prod_alt_desc').show(20, False)
    
#     # where they sold
#     (item
#      .filter(F.col('fis_week_id').between(201949, 201951))
#      .filter(F.col('prod_code').isin('012231304'))
#      #.filter(F.col('net_spend_amt') > 0)
#      .select('store_code')
#      .drop_duplicates()
#     ).show(10, False)
    # Test save object
#     import pickle
    
#     lfl = ds.LFLStore(201901, 201902, 'across')
    
#     with open(os.path.join(config_path, 'object_store.pkl'), 'wb') as f:
#               pickle.dump(lfl, f)

    # Check water
#     txn = ds.itemFct('fis_week_id', 201815, 201951, 'hdet', cust_seg=False)
#     pro_path = os.path.join(master_path, 'HDE_HDET', '5_Water', 'HDET', 'config')
    
#     pro_prod_list = pd.read_csv(os.path.join(pro_path, 'prod_list.csv'), dtype={'prod_code':object})['prod_code'].values.tolist()
#     pro_date_list = pd.read_csv(os.path.join(pro_path, 'filter_date.csv'))['date_id'].values.tolist()
    
#     pro_txn = (
#         txn.df
#         .filter(F.col('date_id').isin(pro_date_list))
#         .filter(F.col('prod_code').isin(pro_prod_list))
#     )
    
#     (pro_txn
#      .groupBy('date_id')
#      .agg(F.countDistinct('transaction_fid'), F.countDistinct('card_id'))
#      .orderBy('date_id')
#     ).show(10, False)

