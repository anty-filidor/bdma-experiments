import networkx as nx
import network_diffusion as nd


def get_er_net():
    layer = nx.erdos_renyi_graph(n=1000, p=0.1)
    return nd.MultilayerNetwork.from_nx_layer(layer, ["awareness", "contagion"])


def get_sf_net():
    _layer = nx.scale_free_graph(n=1000, alpha=0.41, beta=0.54, gamma=0.05)
    layer = nx.DiGraph(_layer)
    return nd.MultilayerNetwork.from_nx_layer(layer, ["awareness", "contagion"])


def get_aucs_network():
    net = nd.MultilayerNetwork.from_mpx(file_path="aucs.mpx")
    net.layers.pop('coauthor')
    net.layers.pop('lunch')
    net.layers.pop('leisure')
    net.layers["contagion"] = net.layers.pop("work")
    net.layers["awareness"] = net.layers.pop("facebook")
    return net.to_multiplex()
