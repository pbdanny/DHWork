# Seasonality index
# From core_model 1.9.3
# ti_thailand
# ------------------------

# python 3.7

import sys
sys.path.insert(0, '/nfs/science/shared/ipythonNotebooks/thanakb/dh_python_fn/')
from dhSparkInit import *

from dhMail import sentMeMail
from dhLog import getLog
import dhSpark as ds

import pandas as pd
import numpy as np

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

# -----------------
# Utility function
# -----------------

def get_week_epoch_num(fis_week_id):
    """
    Get week epoch number from fis_week_id
    """
    week_epoch_num = date.filter(F.col('fis_week_id') == fis_week_id).select('fis_week_of_epoch_num').take(1)[0][0]

    return week_epoch_num

def get_prod_lv_wkly_kpi(prod_lv_code, start_wk, end_wk, ma_wk_range):
    """
    Get product level units, sales for moving ma analysis
    The code will also check if the weekly kpi available every week

    Parameter :-
    prod_lv_code (string) : hiearchy for seasonal index cal
    start_wk (int) : start week that get wkly index
    end_wk (int) : end week that get wkly index
    ma_wk_range (list) : [ no of leading week , no of lagging week]
    ** Lagging week, will include self week as center of MA index

    Output :-
    (spark_frame)
    """
    if type(ma_wk_range) is list:
        LEAD_WK_NUM = np.abs(ma_wk_range[0])
        # lag week add 1 for center of MA index
        LAG_WK_NUM = np.abs(ma_wk_range[1]) + 1

    else:
        print('Use MA wk parameter with list() [LEAD_WK, LAG_WK]')

    # KPIs start week = start index period + number of lead week
    start_ma_wk_id = ds.get_back_wk_id(start_wk, back_wk= LEAD_WK_NUM)
    # KPIs end week = start index period + number of lag week
    end_ma_wk_id = ds.get_back_wk_id(end_wk, back_wk= -LAG_WK_NUM)

    start_ma_wk_epoch = get_week_epoch_num(start_ma_wk_id)
    end_ma_wk_epoch = get_week_epoch_num(end_ma_wk_id)

    # number of week priod for ma
    num_wk_ma = end_ma_wk_epoch - start_ma_wk_epoch + 1

    print('MA of week {} - {}'.format(start_ma_wk_id, end_ma_wk_id))
    print('MA Lead {} wk, Lag {} wk'.format(LEAD_WK_NUM, LAG_WK_NUM))
    print('MA need at least {} wk data point'.format(num_wk_ma))

    # prod level for aggregate
    prod_lv = (
        prod
        .filter(F.col('prod_hier_l50_code').isin(key_division_hier_l50_code))
        .select('prod_code', prod_lv_code)
        .drop_duplicates()
    )

    # prod_code, fis_week_id, units, sales
    prod_lv_wkly_kpi = (
        item
        .filter(F.col('fis_week_id').between(start_ma_wk_id, end_ma_wk_id))
        .filter(F.col('net_spend_amt') > 0)
        .join(broadcast(prod_lv), 'prod_code')
        .withColumn('units',
                    F.when(F.col('weight_uom_qty') > 0, F.col('weight_uom_qty'))
                    .when(F.col('weight_uom_qty').isNull(), F.col('item_qty'))
                    .otherwise(F.col('item_qty')))
        .select(prod_lv_code, 'fis_week_id', 'units', 'net_spend_amt')

        .groupby(prod_lv_code, 'fis_week_id')

        .agg(F.sum('units').alias('units'),
             F.sum('net_spend_amt').alias('sales'))
    )

    # Remmove product that not have data for whole period
    count_prod_lv_wk = prod_lv_wkly_kpi.groupby(prod_lv_code).agg(F.count('fis_week_id').alias('n_wk'))
    n_count_prod_lv = count_prod_lv_wk.count()

    prod_lv_whole_period = count_prod_lv_wk.filter(F.col('n_wk') >= num_wk_ma).select(prod_lv_code)
    n_prod_lv_whole_perod = prod_lv_whole_period.count()

    print('Analysis level at {}. Total {} subjects, satisfy MA range {}'.format(prod_lv_code, n_count_prod_lv , n_prod_lv_whole_perod))

    prod_lv_wkly_kpi_whole_period = prod_lv_wkly_kpi.join(broadcast(prod_lv_whole_period), prod_lv_code)

    return prod_lv_wkly_kpi_whole_period

# ------------------
# Function - Median
# ------------------

def find_median(values_list):
    """
    Median from list of data
    """
    try:
        median = np.median(values_list) #get the median of values in a list in each row
        return round(float(median),2)
    except Exception:
        return None #if there is anything wrong with the given values

median_finder = F.udf(find_median, T.FloatType())

# ---------------
# Seasonality Fn
# ---------------

def seasonality_index(data_frame, kpis, time_frame, prod_lv, mv_avg=[-26,25]):
        """
        Returns moving average of specified columns, base on census 1 method
        ** Not support 52x2 MA

        Parameter :-
        data_frame (sparkFrame) : data frame with kpi
        kpis (list) : list of column name for ma kpi calculation, usually 'units', 'sales'
        time_frame (str) : col name for time frame to calculated as MA, usually 'fis_week_id'
        prod_lv (str, default = '') : col name for product level to calculate MA
        ma_avg (list, defautl = [-26,25]) : 52MA with lead 26 weeks & 1 self week & lag 25 week

        Output :-
        out (sf) : with column
            prod_lv :
            period_no : last 2 digits of time_frame key, if time_frame key is 'fis_week_id' -> week no.
            med_idx_units : median of each year seasonality index for units
            med_idx_sales : median of each year seasonality index for sales
        """
        # casting the parameter string to int
        start, end = int(mv_avg[0]), int(mv_avg[1])

        # MA period number = start + end + 1 (self)
        period_lenght = np.abs(start) + np.abs(end) + 1
        print('MA by {} with period {}'.format(prod_lv, period_lenght))

        print("moving average will be calculated for :")
        print(kpis)

        # Creating the window for the given columns
        # MA windows
        window = Window.partitionBy(str(prod_lv)).orderBy(time_frame).rowsBetween(start, end)
        # Whole time frame windows
        window2 = Window.partitionBy(str(prod_lv)).orderBy(time_frame)

        # find moving average, index of the specified columns
        for kpi in kpis:
            # calculate moving average of the specified columns
            data_frame = data_frame.withColumn('ma_{}'.format(kpi), F.avg(F.col(kpi)).over(window))
            data_frame = data_frame.withColumn('rank', F.rank().over(window2))
            data_frame = data_frame.withColumn('idx_{}'.format(kpi), F.col(kpi)/F.col('ma_{}'.format(kpi)))

        # Trim period rank < 26 and rank > 25 from that ma, idx data from not full 52 windows range
        max_rank = data_frame.select(F.max('rank')).take(1)[0][0]
        min_rank_bound = -start
        max_rank_bound = max_rank - end + 1
        data_frame = data_frame.filter(F.col('rank') > min_rank_bound).filter(F.col('rank') < max_rank_bound)

        # Create period_no for final index
        data_frame = data_frame.withColumn('period_no', F.substring(F.col(time_frame), 5, 2))

        # create blank dict for store output of each kpi
        out_dict = dict()
        # Median of each products idx as final index
        for kpi in kpis:
            out_dict[kpi] = (
                data_frame
                .groupby(prod_lv, 'period_no')
                .agg(F.collect_list('idx_{}'.format(kpi)).alias('idx_list'))
                .withColumn('med_idx_{}'.format(kpi), median_finder('idx_list'))
            )

        # combine all kpi in to one data frame
        for i, (k, v) in enumerate(out_dict.items()):
            # first sf in dict copy only use columns
            if i == 0:
                out = v.select(prod_lv, 'period_no', 'med_idx_{}'.format(k))
            # next sf, create v_sel as select only coulumn use for join
            else:
                v_sel = v.select(F.col(prod_lv).alias('prod_lv_j'),
                                 F.col('period_no').alias('period_no_j'),
                                 'med_idx_{}'.format(k))
                out = out.join(v_sel, [(out[prod_lv] == v_sel['prod_lv_j']),
                                       (out['period_no'] == v_sel['period_no_j'])])
                # remove redundant columns after join
                all_cols = out.columns
                all_cols.remove('prod_lv_j')
                all_cols.remove('period_no_j')
                out = out.select(all_cols)

        print("Moving avg completed")

        return out

if __name__ == "__main__":

    # Define parameter for analysis
    kpis = ['units', 'sales']
    prod_lv = 'prod_hier_l10_code'
    time_frame = 'fis_week_id'
    mv_avg=[-26, 25]

    # Get weekly KPIs
    kpi = get_prod_lv_wkly_kpi(prod_lv, 201713, 201912, mv_avg)
    kpi_df = ds.to_pandas(kpi)
    kpi_df.to_excel(os.path.join(result_path, 'seasonality_kpi.xlsx'), index = False)

    # Calculate seasonality index

    # Spark use current location as 1 row
    # rowsBetween -26, 25 = 26 previous row + 1 current row + 25 preceeding
    # which equal 52 rows

    data_frame = kpi
    sea_idx = seasonality_index(data_frame, ['units','sales'], time_frame, prod_lv)
    ds.to_pandas(sea_idx).to_excel(os.path.join(result_path, 'l10_52MA_sea_idx.xlsx'), index = False)

# -----------
# Finalizing
# -----------
print('========================')
print('Data Preparation finish')
print('========================')
sentMeMail('Step {} finish'.format(__file__))
ss.stop()