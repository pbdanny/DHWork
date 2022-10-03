# Seasonality index
# From core_model 1.9.3
# ti_thailand
# ------------------------

# Python 2.7

import sys
sys.path.insert(0, '/nfs/science/shared/ipythonNotebooks/thanakb/dh_python_fn/')
from dhSparkInit import *

from dhMail import sentMeMail
# from dhLog import getLog
import dhSpark as ds

import pandas as pd
import numpy as np

egg = '/nfs/science/shared/lib/analyst_solutions/eggs/core_modulesv1.9.2.egg'
sc.addPyFile(egg)
egg = '/nfs/science/shared/lib/mercury/tesco_th/dunnhumby-1.0-py2.7.egg'
sc.addPyFile(egg)
egg = '/nfs/science/shared/lib/mercury/tesco_th/tesco_th-1.0-py2.7.egg'
sc.addPyFile(egg)

import core_modules
import dunnhumby
import tesco_th

client = tesco_th.client.Client('dev', sc, sqlContext)

from core_modules import seasonality

seasonality.get_seasonality(client, start_week=201820, end_week=201821, data_week_end=201821, category='prod_hier_l50_code', measures=['units'], weeks=104, mv_avg=[-2,1])