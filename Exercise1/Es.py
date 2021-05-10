import pandas as pd
import pm4py
import os

# importing BPMN and generating Petri Net

from pm4py.objects.conversion.bpmn import converter as bpmn_converter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter


bpmn = pm4py.read_bpmn(os.path.join('diagram-3.bpmn'))

net, initial_marking, final_marking = bpmn_converter.apply(bpmn)

pnml_exporter.apply(net, initial_marking, "diagram-3.pnml")

# Generating a synthetic log

from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.conversion.log import converter as log_converter


simulated_log = simulator.apply(net, initial_marking, variant=simulator.Variants.BASIC_PLAYOUT, parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: 10})

#print(simulated_log)

dataframe = log_converter.apply(
simulated_log, variant=log_converter.Variants.TO_DATA_FRAME)
dataframe.to_csv('daigram-3_log.csv')

# Run Conformance Checking

from pm4py.objects.log.util import dataframe_utils
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

log_csv = pd.read_csv('daigram-3_log_anomalous.csv', sep=',')
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
log_csv = log_csv.sort_values('time:timestamp') # sort by the timestamp column
#log_csv.rename(columns={'Case ID': 'case:concept:name', 'Start Timestamp': 'start_timestamp', 'Complete Timestamp':'time:timestamp', 'Activity': 'concept:name'}, inplace=True) #change the name to a colum
parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'} # identify the case_id_key name (if not change it will simply be the mane of the coloumn)
event_log = log_converter.apply(log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)


replayed_traces = token_replay.apply(event_log, net, initial_marking, final_marking)

print(replayed_traces)
