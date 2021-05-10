## 
## (i) import pnml file (ii) Visualize Petri Net (iii) Compute Reachability Graph (iv) Visualize Reachability Graph 
##

import os
import pandas as pd
import pm4py

from pm4py.objects.petri_net.importer import importer as pnml_importer
net, initial_marking, final_marking = pnml_importer.apply(os.path.join("scenario1.pnml"))

from pm4py.visualization.petri_net import visualizer as pn_visualizer
gviz = pn_visualizer.apply(net, initial_marking, final_marking)
pn_visualizer.view(gviz)

from pm4py.objects.petri_net.utils import reachability_graph

ts = reachability_graph.construct_reachability_graph(net, initial_marking)

from pm4py.visualization.transition_system import visualizer as ts_visualizer

gviz = ts_visualizer.apply(ts, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "svg"})
ts_visualizer.view(gviz)
