import networkx as nx
from pgmpy.readwrite import BIFReader
from graphviz import Digraph
import os


# print(os.environ['PATH'])
# exit(0)

def plot(net_name):
    data_path = os.path.dirname(os.path.dirname(__file__)) + "\\data\\" + net_name + "\\"
    print(data_path)
    # exit(0)

    # reader = BIFReader('data/network.bif')  #network
    # reader = BIFReader('data/dataset/alarm.bif')  #Alarm
    # reader = BIFReader('../bn/child.bif')  # Large BN
    # reader = BIFReader('data/dataset/andes.bif')  #Very large BN
    # !rm friends.bif
    # reader = BIFReader(net_name + ".bif")
    print(data_path + net_name + ".bif")
    reader = BIFReader(data_path + net_name + ".bif", n_jobs=1)
    network = reader.get_model()

    G = Digraph('network')
    # nodes = network.nodes()
    # print(nodes)
    # exit(0)

    # nodes_sort = list(nx.topological_sort(network))
    # print('nodes_num:', len(nodes_sort))
    # print('nodes_sort:', nodes_sort)
    # exit(0)

    edges = network.edges()
    # print('\nedges_num:', len(edges))
    # print('\nedges:', edges)

    for a, b in edges:
        G.edge(a, b)
    # var_card = {node: network.get_cardinality(node) for node in nodes_sort}
    # print('var_card:', var_card)
    # var_card=dict(zip(cpd.variables, cpd.cardinality))

    G.render(data_path + net_name + ".gv", view=False)
    # G
    # print(len(nodes))



# bn_name = ['heart']
# bn_name = ['hepar2']
# bn_name = ['bone']
bn_name = ['munin1']
for net_name in bn_name:
    plot(net_name)