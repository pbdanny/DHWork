
import sys
sys.path.insert(0, '/nfs/science/shared/ipythonNotebooks/thanakb/dh_python_fn/')
from dhSparkInit import *

from dhMail import sentMeMail
from dhLog import getLog
import dhSpark as ds
# import dhExadata as dx

import pandas as pd
import numpy as np

import os

# -------------------------
# Shopping cycle
# -------------------------

def shopping_cycle(txn):
    """
    For calculate shopping cycle from transaction provided with column 'card_id'
    and 'date_id'
    
    Only Clubcard, with transaction days >= 2
    Shopping cycle calculate from Median of day next purchased
    Overall group calculate from Median shopping cycle in this group
    
    Parameter :- 
    txn (spark data frame) : transaction with column 'card_id', 'date_id'
    
    Ouput (tuple) : (out_df, overall_cycle)
    out_df (spark data frame) : Output card_id, median_day_cycle
    overall_cycle (int) : Overall shopping cycle of this group
    
    """
    shopper_day_diff = (
        txn
        .filter(F.col('card_id').isNotNull())
        .select('card_id', 'date_id')
        .drop_duplicates()
        .select('card_id', 'date_id',
                F.lead('date_id')
                .over(Window.partitionBy('card_id').orderBy('date_id'))
                .alias('next_date_id'))
        .filter(F.col('next_date_id').isNotNull())
        .withColumn('day_next_purchase', F.datediff(F.col('next_date_id'), F.col('date_id')))
    )
    # Function to find the medion from set of date
    median_udf = F.udf(lambda x: float(np.median(x)))
    
    # median os shoppin day diff
    shopping_cycle = (
        shopper_day_diff
        .groupBy('card_id')
        .agg(F.collect_list('day_next_purchase').alias('list_day_next_purchase'))
        .withColumn('median_day_cycle', median_udf('list_day_next_purchase'))
    )
    
    # Overall group shopping cycle
    overall_cyc = float(shopping_cycle.select(median_udf('median_day_cycle')).take(1)[0][0])
    # print result of overall customer group shopping cycle
    print('Overall customer shopping cycle (day) : {}'.format(overall_cyc))
    
    # return output sf, and overall group cycle
    out_sf = shopping_cycle.select('card_id', 'median_day_cycle')
    return out_sf, overall_cyc

# -------
# Main
# -------

if __name__ == '__main__':

    # ------------------------
    # test product
    # OLEEN Palm oil
    # prod_code : 007547285
    # ------------------------

    oleen1000cc_code = '002614359'

    analysis_store_format_code = ['1','2','3']

    analysis_store_code = store.filter(F.col('format_code').isin(analysis_store_format_code)).select('store_code').drop_duplicates()

    itm = ds.itemFct('fis_week_id', 201925, 201936, 'hde')

    shopper_day = (
        itm
        .df
        .filter(F.col('card_id').isNotNull())
        .filter(F.col('prod_code') == oleen1000cc_code)
        .join(broadcast(analysis_store_code), 'store_code')
        .select('card_id', 'date_id')
        .drop_duplicates()
    )
    
    # Get shopping by card_id
    oleen_shopping_cycle, _ = shopping_cycle(shopper_day)
    oleen_shopping_cycle_df = ds.to_pandas(oleen_shopping_cycle)    
    # convert object to float
    oleen_shopping_cycle_df['median_day_cycle'] = oleen_shopping_cycle_df['median_day_cycle'].astype(float)
    
    # %mathplotlib inline
    # Histogram of shopping cycle
    oleen_shopping_cycle_df['median_day_cycle'].hist(bins = 20)
    
    # Calculate accumulate sum for segment 
    count_card_by_median_day = oleen_shopping_cycle_df.groupby(['median_day_cycle'])[['card_id']].count()
    count_card_by_median_day['accu_n_card'] = count_card_by_median_day.cumsum()
    count_card_by_median_day['accu_n_card'].plot('line')
    
ss.stop()