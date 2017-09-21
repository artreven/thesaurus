import rdflib
from scipy import stats, spatial
import numpy as np

import networkx as nx

import pp_api
from thesaurus import Thesaurus

swc_onto = rdflib.Namespace('http://schema.semantic-web.at/ppcm/2013/5/')


def get_corpus_distsim_score(the, sparql_endpoint, cpt_cooc_graph):
    cpt_list = list(map(str, set(the.get_all_concepts()) -
                             {the.top_uri.toPython()}))
    # get lin distances
    lin_dists = []
    for i in range(len(cpt_list)):
        cpt1 = cpt_list[i]
        for j in range(i+1, len(cpt_list)):
            cpt2 = cpt_list[j]
            lin_dists.append(the.get_lin_similarity(cpt1, cpt2))
    # get cooc distances
    cooc_dist_mx = pp_api.query_cpt_cooc_scores(sparql_endpoint,
                                                cpt_cooc_graph)
    cooc_dists = []
    for i in range(len(cpt_list)):
        cpt1 = cpt_list[i]
        for j in range(i + 1, len(cpt_list)):
            cpt2 = cpt_list[j]
            cooc_dists.append(
                cooc_dist_mx[cpt1][cpt2]
                if cpt1 in cooc_dist_mx and cpt2 in cooc_dist_mx[cpt1]
                else 0
            )
    slope, intercept, r_value, p_value, std_err = stats.linregress(lin_dists,
                                                                   cooc_dists)
    return slope, intercept, r_value, p_value, std_err, lin_dists, cooc_dists


def get_cooc_graph(sparql_endpoint, cpt_cooc_graph):
    cooc_dist_mx = pp_api.query_cpt_cooc_scores(
        sparql_endpoint, cpt_cooc_graph
    )
    G = nx.Graph()

    nodes = list(cooc_dist_mx.keys())
    G.add_nodes_from(nodes)
    for c1 in cooc_dist_mx:
        for c2 in cooc_dist_mx[c1]:
            G.add_edge(c1, c2, weight=cooc_dist_mx[c1][c2])
    return G


def get_broader_transitive_graph(the):
    G = the.get_nx_graph()
    G_closed = nx.transitive_closure(G)
    G_related = the.get_nx_graph(use_related=True)
    related_edges = [edge
                     for edge in G_related.edges_iter(data='relation', default=None)
                     if edge[2] is not None]
    for edge in related_edges:
        G_closed.add_edge(edge[0], edge[1], relation=edge[2])
    return G_closed


if __name__ == '__main__':
    import os
    from pp_api.server_data.custom_apps import pid, server, \
        corpus_id, sparql_endpoint
    import plotting

    auth_data = tuple(map(os.environ.get, ['pp_user', 'pp_password']))

    corpusgraph_id, termsgraph_id, cpt_occur_graph_id, cooc_graph = \
        pp_api.get_corpus_analysis_graphs(corpus_id)

    the_path = 'misc/the.n3'
    the = Thesaurus.get_the(
        the_path=the_path, sparql_endpoint=sparql_endpoint,
        cpt_freq_graph=cpt_occur_graph_id, server=server,
        pid=pid, auth_data=auth_data
    )
    print(len(the))

    G = the.get_nx_graph()
    print(len(G.edges()))
    print(len(G.nodes()))

    G = the.get_nx_graph(use_related=True)
    print(len(G.edges()))
    print(len(G.nodes()))
    # print(G.edges(data=True))

    G = get_broader_transitive_graph(the)
    print(len(G.edges()))
    print(len(G.nodes()))

    G2 = get_cooc_graph(sparql_endpoint, cpt_occur_graph_id)
    print(len(G2.edges()))
    print(len(G2.nodes()))
