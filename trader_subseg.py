# ---------------------------------------
# Trader Subsegment - Trail
# Create needstage
# ---------------------------------------

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

import time
start_time = time.monotonic()


# In[2]:


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
    data_path = os.path.join(master_path, 'data')
    
    nfs_path = '/nfs/science/ti_thailand/thanakb/factor_segment'
    
    # ------------------------
    # Parameter for analysis
    # ------------------------
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(os.path.join(config_path, 'EE_BDA.conf'))

    # project
    PROJ = config['project']['proj']

    # store format
    STR_FORMAT = config['store_format']['str_format']

    # num partition size
    # data 6 mth ~ 60MB fit 1 partition size
    # Usually, set 40 for pararallism (10 executor x 4 core)
    SAVE_PARTITION_SIZE = int(config['hive_config']['save_partition_size'])

    # Products
    PROD_SEED_FILE = config['products']['promo_seed_file']

    # Promo Year
    period = config['period']
    ty_dur_str_date_id = period['ty_dur_str_date_id']
    ty_dur_end_date_id = period['ty_dur_end_date_id']

    # get week id of date id param
    ty_dur_str_wk_id = ds.get_week_id(ty_dur_str_date_id)
    ty_dur_end_wk_id = ds.get_week_id(ty_dur_end_date_id)
    


# In[4]:


# --------------------------
# Load transaction and flag
# --------------------------
itm = ss.table('tesco_th_analyst.thanakb_{}_itm'.format(PROJ))

# subclass code
l10_code = (
    prod
    .filter(F.col('prod_hier_l50_code').isin(key_division_hier_l50_code))
    .select('prod_code', 'prod_hier_l10_code')
    .drop_duplicates()
)

# map subclass code
itm = itm.join(l10_code, 'prod_code')

# visit , sales by card x subcl
card_subcl_met = (
    itm
    .groupBy('card_id', 'prod_hier_l10_code')
    .agg(F.countDistinct('transaction_fid').alias('subcl_visits'), F.sum('net_spend_amt').alias('subcl_sales'))
)

# visit, sales by card
card_met = (
    itm
    .groupBy('card_id')
    .agg(F.countDistinct('transaction_fid').alias('visits'), F.sum('net_spend_amt').alias('sales'))
)

# compute pct_visit, pct_sales
card_subcl_pct = card_subcl_met.join(card_met, 'card_id')
card_subcl_pct = (
    card_subcl_pct
    .withColumn('pct_visits', F.col('subcl_visits')/F.col('visits'))
    .withColumn('pct_sales', F.col('subcl_sales')/F.col('sales'))
    .withColumn('subcl', F.concat(F.lit('s'), F.col('prod_hier_l10_code')))
)

# select card, subcl, pct_visit
need_met = card_subcl_pct.select('card_id', 'subcl', 'pct_visits')

need_met.coalesce(40).write.mode('overwrite').saveAsTable('tesco_th_analyst.thanakb_{}_need_met'.format(PROJ))
print('Create combine itmFct hive table at : tesco_th_analyst.thanakb_{}_need_met'.format(PROJ))


# In[6]:


# --------------------------------
# Load card x subcl - pct visit
# For needstate 
# --------------------------------
need_met = ss.table('tesco_th_analyst.thanakb_{}_need_met'.format(PROJ))

# pivot data
need_met_pivot = need_met.groupBy('card_id').pivot('subcl').agg(F.first('pct_visits')).fillna(0)
features_name = need_met_pivot.columns[1:]

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# turn data into column vector
features_data = need_met_pivot.columns[1:]
vec_assm = VectorAssembler(inputCols=features_data, outputCol='v_feats')
features_vec = vec_assm.transform(need_met_pivot).select('v_feats')


# In[7]:


# -----------------------
# Create correlation df
# -----------------------
corr = Correlation.corr(features_vec, 'v_feats')
corr_mat = corr.collect()[0][0]
corr_df = ss.createDataFrame(corr_mat.toArray().tolist())
corr_df_name = corr_df.toDF(*features_name)


# In[30]:


corr_df_name.coalesce(1).write.mode('overwrite').parquet('factor_segment.folder/correlation_df.parquet')


# In[30]:


# data to nfs
ds.to_pandas(corr_df_name).to_csv(os.path.join(nfs_path, 'corr_df.csv'))


# In[3]:


# -----------
# Spark PCA
# -----------
corr_df_name = ss.read.parquet('factor_segment.folder/correlation_df.parquet')


# In[4]:


features_name = corr_df_name.columns
from pyspark.ml.feature import PCA, PCAModel
from pyspark.ml.feature import VectorAssembler    


# In[11]:


vec_assm = VectorAssembler(inputCols=features_name, outputCol='features')
corr_vec = vec_assm.transform(corr_df_name)
pca = PCA(k=4, inputCol='features', outputCol="pca_features")
pca_model = pca.fit(corr_vec)    


# In[12]:


# Explained varience
pca_model.explainedVariance    


# In[3]:


# PC metrix 
pca_model.pc

pca_model.write().save('factor_segment.folder/pca_corr_vec.model')
pca_model = PCAModel.read().load('factor_segment.folder/pca_corr_vec.model')


# In[1]:


# ----------------
# Python 3 only
# PCA and Factor analysis 
# Statsmodel
# ----------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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
data_path = os.path.join(master_path, 'data')

nfs_path = '/nfs/science/ti_thailand/thanakb/factor_segment'


# In[6]:


corr_mat = pd.read_csv(os.path.join(nfs_path, 'corr_df.csv'), index_col=0)


# In[7]:


# ---------------
# Factor analysis
# ---------------
from statsmodels.multivariate.factor import Factor
from statsmodels.multivariate.factor_rotation import rotate_factors

factor_model = Factor(corr = corr_mat.values, n_factor=11)


# In[8]:


factor_fit = factor_model.fit()    


# In[9]:


# Screee plot
# fig, ax = plt.subplots(figsize=(20, 10))
fig = factor_fit.plot_scree()
fig.set_size_inches(9, 5)
fig.savefig(os.path.join(figure_path, 'test2png.png'), dpi=100)


# In[10]:


# Eigenvals
np.sum(np.cumsum(factor_fit.eigenvals)/factor_fit.eigenvals.sum() < 0.8)
N_PC = 12 # or 15


# In[11]:


# Correct loadings
factor_fit.loadings


# In[12]:


factor_fit.get_loadings_frame()


# In[13]:


# try rotation, inplace
factor_fit.rotate('varimax')


# In[14]:


# matched wtih book
factor_fit.loadings


# In[15]:


factor_fit.uniqueness


# In[16]:


factor_fit.get_loadings_frame()


# In[30]:


# Get rotated loading as dataframe
pc11_rotated = factor_fit.get_loadings_frame(style = 'strings')


# In[20]:


# save factor
pc11_rotated.to_csv(os.path.join(result_path, 'pc11_rotated.csv'), index = True)


# In[25]:


# get varname from original corr_mat
col_name = corr_mat.columns


# In[34]:


# Resort pc by index name
pc11_rotated.sort_index()

# set pc index by stored varname
pc11_rotated.index = col_name

# save named pc
pc11_rotated.to_csv(os.path.join(result_path, 'pc11_rotated_name.csv'), index = True)


# In[2]:


df = pd.read_csv(os.path.join(result_path, 'pc11_rotated_name.csv'), index_col = 0)


# In[2]:


# --------------------------
# Check the PC Factor
# with data description
# --------------------------

df = pd.read_csv(os.path.join(result_path, 'pc11_rotated_name.csv'), index_col = 0)


# In[4]:


# for PC1
pc1_df = df.loc[~df['factor 5'].isna(), ['factor 5']]

pc1_subcl = pc1_df.index


# In[97]:


pc1_subcl_df = pc1_subcl.str[1:].to_frame()


# In[98]:


pc1_subcl_df.columns = ['prod_hier_l10_code']
pc1_subcl_df.to_csv(os.path.join(result_path, 'df.csv'), index=False)


# In[100]:


ds.copy_file_to_hdfs(os.path.join(result_path, 'df.csv'))


# In[101]:


load_schema = T.StructType([T.StructField('prod_hier_l10_code', T.StringType(), True)])
l10 = ss.read.csv('df.csv', schema=load_schema, header=True)


# In[ ]:


l10.show()


# In[93]:


prod.select('prod_hier_l10_code', 'prod_hier_l10_desc').drop_duplicates().join(l10, 'prod_hier_l10_code').show()

