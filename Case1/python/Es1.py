import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.filtering.log.variants import variants_filter

log_csv = pd.read_csv("../CallCenterLog.csv", sep=',')
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
log_csv = log_csv.sort_values('Start Date')
log_csv.rename(columns={'Case ID': 'case:concept:name', 'Start Date': 'time:timestamp', 'Activity': 'concept:name', 'Resource': 'org:resource'}, inplace=True) #change the name to a colum
parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'} # identify the case_id_key name (if not change it will simply be the mane of the coloumn)
event_log = log_converter.apply(log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

#print(len(event_log))
#print(event_log[0])

count_product = log_csv.groupby('Topic').agg(\
Product=('Topic', 'count'),\
# Multiple aggregations of the same column using pandas ...
FirstOccurence=('time:timestamp', lambda x: x.min()),
LastOccurence=('time:timestamp', lambda x: x.max()),
Endurance=('time:timestamp', lambda x: x.max() - x.min()),
)
print(count_product)

count_activity = log_csv.groupby('concept:name').agg({\
'concept:name': 'count'})

print(count_activity)

## List variants, variants_filter returns a dict type
variants = variants_filter.get_variants(event_log)
# number of events, cases, and variants included in the list
print('Events:', len(log_csv), '- Cases: ', len(event_log),'- Variants:', len(variants))

## Filter most common variants but returns the cases without any reference to the variants
filtered_log = variants_filter.filter_log_variants_percentage(event_log, percentage=0.85)
print(len(filtered_log))
# to get the variants we have to generate the variants again from the filtered_log
variants_filtered = variants_filter.get_variants(filtered_log)
print(len(variants_filtered))
# if we want to generate the variants from the complement (cases not matching the filtering criteria)
filtered_log_compl = variants_filter.apply(event_log, variants_filtered, parameters={variants_filter.Parameters.POSITIVE: False})
variants_filtered_compl = variants_filter.get_variants(filtered_log_compl)
print(len(variants_filtered_compl))















