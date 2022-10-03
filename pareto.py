sys.path.insert(0, '/nfs/science/shared/ipythonNotebooks/thanakb/dh_python_fn/')
from dhSparkInit import *

from dhMail import sentMeMail
from dhLog import getLog
import dhSpark as ds
import dhExadata as dx

import pandas as pd
import numpy as np
import csv

# initialized logger
logger = getLog()

# ------------------------
# Defined param
# ------------------------
START_WK_ID = 201832
END_WK_ID = 201852
LEVEL_OF_ANALYSIS = 'prod_code'
KEY = 'transaction_fid' # 'card_id'
N_TOP = 60

# Print parameter
print('\n' + '-'*15 + ' PARETO ' + '-'*15)

print('Period of analysis : {} - {}'.format(START_WK_ID, END_WK_ID))
print('Key of analysis : {}'.format(KEY))
print('Product level of anlysis : {}'.format(LEVEL_OF_ANALYSIS))
print('Top {} rank'.format(N_TOP))

# Get data and save
# itm = ds.itemFct('fis_week_id', START_WK_ID, END_WK_ID).df
# Get data for Super Starline
# 201808 - 201812 , CC only, End customer
itm = ds.itemFct('fis_week_id', START_WK_ID, END_WK_ID).df.filter(F.col('card_id').isNotNull()).filter(~F.col('kptr_seg').isin('Trader','trader'))
# Create prod group
analysis_prod = pd.read_excel('./Star line list HEDT and Express.xlsx', sheet_name=1)
# analysis_prod['count_digits'] = analysis_prod.prod_style_code.map(lambda x : len(str(x)))
# sum(analysis_prod.count_digits == 9)
analysis_prod_df = ss.createDataFrame(analysis_prod).select('prod_style_code').distinct()

# Use prod_hier_l20 (Class) as product group for analysis
analysis_prod_code = (prod
 .filter(F.col('prod_hier_l50_code').isin(key_division_hier_l50_code))
 .join(analysis_prod_df, 'prod_style_code')
 .select(LEVEL_OF_ANALYSIS)
).dropDuplicates()

# Agg data, if key is card id -> select onty cc data
if KEY is 'card_id':
    token_agg = (itm
    .filter(F.col('card_id').isNotNull())
    .join(hdetex_store_id, 'store_id')

    .join(analysis_prod_code, 'prod_code')
    .groupBy(KEY)
    .agg(F.collect_set(LEVEL_OF_ANALYSIS).alias('set_cat'))
    )

else:
    token_agg = (itm
    .join(hdetex_store_id, 'store_id')

    .join(analysis_prod_code, 'prod_code')
    .groupBy(KEY)
    .agg(F.collect_set(LEVEL_OF_ANALYSIS).alias('set_cat'))
    )

# savet data to hive
token_agg.coalesce(100).write.mode('overwrite').saveAsTable('tesco_th_analyst.thanakb_pareto_token_agg')

# --------------------
# Pareto Part
# --------------------

# Load agg itm
token_agg = ss.table('tesco_th_analyst.thanakb_pareto_token_agg')
token_agg.persist()

# list of products
analysis_prod_code = (token_agg
 .withColumn('explode_set_cat', F.explode(F.col('set_cat')))
 .select('explode_set_cat')
 .dropDuplicates()
)

analysis_prod_list = ds.to_pandas(analysis_prod_code)['explode_set_cat'].to_list()

# Fn create F.array_contain() for prod in list
def get_combine_array_contain_fn(ls):

    for i, l in enumerate(ls):
        if i == 0:
            cond = F.array_contains('set_cat', l)
        else:
            cond = cond | F.array_contains('set_cat', l)

    return cond

# Fn get token count by of prod list defined
def get_key_count_prod_list(token_agg, prod_list):

    combine_fn = get_combine_array_contain_fn(prod_list)

    # print('Check condition on {}'.format(combine_fn))

    kpi =     (token_agg
     .filter(combine_fn)
    ).count()

    return kpi

# From exclude best list from all prod list to create candidat prod list
# loop combine candidate prod + best prod to find the new best prod

def count_key_best_and_candidate_list(token_agg, best_list, prod_list):

    # From all prod list, exclude best list -> candidate list
    best_prod_set = set(best_list)
    all_prod_set = set(prod_list)
    candidate_prod_set = all_prod_set.difference(best_prod_set)

    # create blank output
    output = pd.DataFrame()

    # Loop each candidate prod set
    for prod in candidate_prod_set:

        # print('Best prod : {}'.format(best_list))
        # print('Candidate Prod : {}'.format(prod))

        # combine best list with each prod to create combination for kpi
        query_prod_list = best_list.copy()
        query_prod_list.append(prod)

        logger.info('Query for candidate {}'.format(query_prod_list))
        # create query prod from best prod + each prod
        count = get_key_count_prod_list(token_agg, query_prod_list)

        qry_df = pd.DataFrame({'candidate_prod':[prod],
                               'count_key':[count]})

        # append kpi to output, assign back to old dataframe
        output = pd.concat([output, qry_df])

    # return output
    output.reset_index(drop = True, inplace = True)
    return output

# Sort count of combined candidate + best to find the new best prod
def sort_candidate_find_next_best(df, best_list):

    # sorted condidate by count of token
    sort_df = df.copy()

    sort_df =     (sort_df
     .sort_values('count_key', ascending = False)
     .reset_index(drop = True)
    )

    # get the top candidate
    new_best_prod = sort_df.iloc[0,0]

    # write query result to files
    with open('./candidate_top5_detials.txt', 'a') as f:
        f.write('\n Best prod list :' + ','.join([str(i) for i in best_list]))
        f.write('\n Top 5 candidate :\n')
        f.write(sort_df.iloc[:6,:].to_string())

    # write best prod kpi to files
    with open('./pareto.csv', 'a') as f:
        sort_df.iloc[:1,:].to_csv(f, header = False, index = False)

    return new_best_prod

# ----------------------------------------------------------------
# MAIN
# 1. Loop all prod_list
# 2. Query kpi of best prod_id + each prod_id
# 3. Find the new best prod
# 4. Append new best prod to best list, save new prod kpi , save top 5 candidate to file
# 5. Mail result, kpi, candidate
# ----------------------------------------------------------------

# remove old report if existed
try:
    os.remove('./candidate_top5_detials.txt')
    os.remove('./pareto.csv')
    logger.info('Delete old report files')
except:
    logger.info('No existing report files')

# Query Prod list
print('Candidate prod_code total : {} skus'.format(len(analysis_prod_list)))

# Initiate header in report
with open('./pareto.csv', 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['prod', 'count_token'])

# Initiate best_list
best_list = []

# N-loop, equal to all prod list
# N_TOP = len(analysis_prod_list)

list_all_prod = analysis_prod_list

# n-loop = number of prod in all_prod_id
for i in range(0, N_TOP):

    # get kpi from combined best list + all prod list
    kpi = count_key_best_and_candidate_list(token_agg, best_list, list_all_prod)

    # Find the best in number of customers
    new_best_prod = sort_candidate_find_next_best(kpi, best_list)

    best_list.append(new_best_prod)

    logger.info('New best prod {}'.format(best_list))

# mail to notify results
sentMeMail('Pareto finish', body = '', file = ['pareto.csv','candidate_top5_detials.txt'])

ss.stop()