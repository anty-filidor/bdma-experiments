import networkx as nx
import network_diffusion as nd
import pandas as pd


def _network_from_pandas(path):
    df = pd.read_csv(path, names=["node_1", "node_2", "layer"])
    net_dict = {l_name: nx.Graph() for l_name in [*df["layer"].unique()]}
    for _, row in df.iterrows():
        net_dict[row["layer"]].add_edge(row["node_1"], row["node_2"])
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=[*net_dict.keys()], network_list=[*net_dict.values()]
    )


def get_er_net():
    layer_c = nx.erdos_renyi_graph(n=1000, p=0.005)
    layer_a = nx.erdos_renyi_graph(n=1000, p=0.05)
    return nd.MultilayerNetwork.from_nx_layers(
        [layer_a, layer_c], ["awareness", "contagion"]
    )


def get_sf_net():
    layer_c = nx.DiGraph(nx.scale_free_graph(n=1000, alpha=0.41, beta=0.54, gamma=0.05))
    layer_a = nx.DiGraph(nx.scale_free_graph(n=1000, alpha=0.41, beta=0.54, gamma=0.05))
    return nd.MultilayerNetwork.from_nx_layers(
        [layer_a, layer_c], ["awareness", "contagion"]
    )


def get_aucs_network():
    return nd.MultilayerNetwork.from_mpx(file_path="data/aucs.mpx")


def get_aucs_2_network():
    net = nd.MultilayerNetwork.from_mpx(file_path="data/aucs.mpx")
    net.layers.pop('coauthor')
    net.layers.pop('work')
    net.layers.pop('leisure')
    net.layers["contagion"] = net.layers.pop("lunch")
    net.layers["awareness"] = net.layers.pop("facebook")
    return net.to_multiplex()


def get_lazega_network():
    return _network_from_pandas(
        "data/Lazega-Law-Firm_4NoNatureNoLoops.edges"
    )
