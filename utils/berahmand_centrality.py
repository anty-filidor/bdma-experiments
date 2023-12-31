"""
This script contains Python implementation of a Berahmand centrailty algorithm.

It was published by: Kamal Berahmand, Asgarali Bouyer, Negin Samadi as: "A new
centrality measure based on the negative and positive effects of clustering 
coefficient for identifying influential spreaders in complex networks" in "
Chaos, Solitons & Fractals",  Volume 110, 2018, Pages 41-54, ISSN 0960-0779,
DOI: https://doi.org/10.1016/j.chaos.2018.03.014.
"""
from typing import Any, Dict, List

import networkx as nx
import network_diffusion as nd


def berahmand_centrality(graph: nx.Graph) -> Dict[int, float]:
    """
    Implementation of Berahmand centrailty.

    Please note, that we are using an implementation of clustering coefficient 
    provided by NetworkX. It returns slightly different values than these 
    presented in the paper, hence results in the ranking slightly differ.

    :param graph: an undirected graph to compute centrality measures for
    :return: a dict keyed by node ids with values standing for corresponding 
        centrality value
    """
    ci_dict = {}

    # get global measures
    K = nx.degree(G)
    CC = nx.clustering(G)

    # compute centrality for each node
    for i in graph.nodes():
        k_i = K[i]
        cc_i = CC[i]
        neigh_1st_i = {*nx.neighbors(G, i)}
        neigh_2nd_i = set().union(*[{*nx.neighbors(G, n)} for n in neigh_1st_i]).difference({i})
        centrality_i = k_i * 1 / (cc_i + 1 / k_i) + sum([CC[n] for n in neigh_2nd_i])
        ci_dict[i] = centrality_i

    return ci_dict


class BerahmandCentralitySelector(nd.seeding.base_selector.BaseSeedSelector):
    """Seed selector based on Berahmand centrailty algorithm."""

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """
        Create a ranking of nodes.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        ranking_dict = berahmand_centrality(graph=graph)
        ranked_nodes = sorted(ranking_dict, key=lambda x: ranking_dict[x], reverse=True)
        if len(ranked_nodes) != len(graph.nodes): raise ValueError
        return ranked_nodes

    def __str__(self) -> str:
        """Return seed method's description."""
        return "Berahmand centrailty-based seed selection method."

    def actorwise(self, net: nd.MultilayerNetwork) -> List[nd.MLNetworkActor]:
        """Compute ranking for actors."""
        return nd.seeding.base_selector.node_to_actor_ranking(super().nodewise(net), net)


if __name__ =="__main__":
    """An example of the centrality according to the original paper."""

    # create a network
    edge_list = [
        [1, 5], [1, 4], [1, 3], [1, 2], [1, 7], [1, 6],

        [32, 6], [32, 31],
        [31, 6], [31, 30], 
        [30, 6], [30, 29],
        [29, 6],

        [28, 5], [28, 24], [28, 27],
        [27, 5], [27, 24], [27, 25], [27, 26],
        [26, 5], [26, 25],
        [25, 5], [25, 24],
        [24, 5],

        [23, 4], [23, 20], [23, 22],
        [22, 4], [22, 21],
        [21, 4], [21, 20],
        [20, 4],

        [19, 3], [19, 18],
        [18, 3], [18, 17],
        [17, 3],

        [16, 2], [16, 12], [16, 15],
        [15, 2], [15, 11], [15, 14],
        [14, 2], [14, 13],
        [13, 2], [13, 12],
        [12, 2], [12, 11],
        [11, 2],

        [10, 7], [10, 8], [10, 9],
        [9, 7], [9, 8],
        [8, 7],
    ]
    G = nx.from_edgelist(edge_list)

    # visualise it
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=1, nrows=1)
    nx.draw(G, pos=nx.spring_layout(G), ax=ax, with_labels=True)
    plt.show()

    # compute the centrailty values
    centralities = berahmand_centrality(G)
    for node in sorted(G.nodes()):
        print(f"{node}: {centralities[node]}")

    # test its integration with a Network DIffusion framework: create a 
    # single-layer multilayer network and then feed a seed selector with it
    multilayer_G = nd.MultilayerNetwork.from_nx_layers([G], ["l1"])
    print(BerahmandCentralitySelector()(multilayer_G))
