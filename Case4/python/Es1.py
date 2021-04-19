import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.filtering.log.variants import variants_filter

df1 = pd.read_csv('./saleslog.csv', parse_dates=['Date'])
df2 = pd.read_csv('./wine_description.csv')

df_merge = pd.merge(df1,df2, left_on='itemDescription', right_on='name')

df_merge['Origin_Type'] = df_merge.apply(lambda row: row.origin +'-'+ row.type, axis=1)

print(df_merge)

log_csv = dataframe_utils.convert_timestamp_columns_in_df(df_merge)
log_csv = log_csv.sort_values('Date') # sort by the timestamp column
# inverting itemDescription and type we can make the trace abstraction lower or higher
log_csv.rename(columns={'Member_number': 'case:concept:name', 'Date': 'time:timestamp', 'Origin_Type': 'concept:name', 'origin': 'org:resource'}, inplace=True) #change the name to a colum
parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'} # identify the case_id_key name (if not changed it will simply be the mane of the coloumn)
event_log = log_converter.apply(log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

## Count solt products

count_product_type = log_csv.groupby('org:resource').agg(\
Type=('org:resource', 'count'),\
# Multiple aggregations of the same column using pandas ...
FirstOccurence=('time:timestamp', lambda x: x.min()),
LastOccurence=('time:timestamp', lambda x: x.max()),
Endurance=('time:timestamp', lambda x: x.max() - x.min()),
)
print(count_product_type)

count_product = log_csv.groupby('concept:name').agg({\
'concept:name': 'count'})

print(count_product)

## List variants 
variants = variants_filter.get_variants(event_log)
# number of events, cases, and variants included in the list
print('Events:', len(log_csv), '- Cases: ', len(event_log),'- Variants:', len(variants))

#print(variants)

## Directly-Follows Graph

from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization

# In principle other variants and parameters are available but they are not suported in the last version of PM4ML 
# https://pm4py.fit.fraunhofer.de/static/assets/api/1.3.4/_modules/pm4py/algo/discovery/dfg/algorithm.html 
dfg = dfg_discovery.apply(event_log, variant=dfg_discovery.Variants.FREQUENCY)
# this setting of the parameters is required to save the result is an SVG file
parameters = {dfg_visualization.Variants.FREQUENCY.value.Parameters.FORMAT: "svg"}
gviz = dfg_visualization.apply(dfg, log=event_log, variant=dfg_visualization.Variants.FREQUENCY, parameters=parameters)

#dfg_visualization.view(gviz)
dfg_visualization.save(gviz, "dfg.svg")














