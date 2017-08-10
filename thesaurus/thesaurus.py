import numpy as np

import rdflib
import requests

import pp_api


class Thesaurus(rdflib.graph.Graph):
    """
    A class for thesauri. Especially useful if you have a corpus for tagging
    with this thesaurus.
    
    In order to get the cumulative concept frequencies you may want to call
     `query_and_add_cpt_frequencies` method.
    """

    own_freq_predicate = rdflib.URIRef(':own_frequency')
    cum_freq_predicate = rdflib.URIRef(':cum_frequency')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uriref = rdflib.URIRef(':T')
        self.add([uriref,
                  rdflib.namespace.SKOS.prefLabel,
                  rdflib.Literal(':T')])
        self.add([uriref,
                  rdflib.namespace.RDF.type,
                  rdflib.namespace.SKOS.Concept])
        self.top_uri = uriref

    def get_all_concepts(self):
        s_pl = self.triples((None,
                             rdflib.namespace.RDF.type,
                             rdflib.namespace.SKOS.Concept))
        return {x[0] for x in s_pl}

    def get_leaves(self):
        brs = {x[2] for x in self.triples((
            None,
            rdflib.namespace.SKOS.broader,
            None
        ))}
        leaves = set(self.get_all_concepts()) - brs
        return leaves

    def get_pref_label(self, uri):
        pl = self.value(rdflib.URIRef(uri), rdflib.namespace.SKOS.prefLabel)
        return pl

    def add_path(self, path):
        prev_uriref = None
        for uri, pref_label in path:
            uriref = rdflib.URIRef(uri)
            self.add([uriref,
                      rdflib.namespace.SKOS.prefLabel,
                      rdflib.Literal(pref_label)])
            self.add([uriref,
                      rdflib.namespace.RDF.type,
                      rdflib.namespace.SKOS.Concept])
            if prev_uriref is not None:
                self.add([
                    uriref,
                    rdflib.namespace.SKOS.broader,
                    prev_uriref
                ])
                self.add([
                    prev_uriref,
                    rdflib.namespace.SKOS.narrower,
                    uriref
                ])
            else:
                self.add([
                    self.top_uri,
                    rdflib.namespace.SKOS.narrower,
                    uriref
                ])
                self.add([
                    uriref,
                    rdflib.namespace.SKOS.broader,
                    self.top_uri
                ])
            prev_uriref = uriref

    def add_frequencies(self, cpt_uri, cpt_freq, path=None):
        if path is None:
            uriref = rdflib.URIRef(cpt_uri)
            brs = {x[2] for x in self.triples((
                uriref,
                rdflib.namespace.SKOS.broader * '*',
                None
            ))}
            path = brs
        else:
            path = {x[0] for x in path}
        old_freq = self.value(
            subject=cpt_uri, predicate=self.own_freq_predicate, default=0
        )
        self.set(
            (cpt_uri,
             self.own_freq_predicate,
             rdflib.Literal(old_freq + cpt_freq))
        )
        for uri in path:
            uriref = rdflib.URIRef(uri)
            old_freq = self.value(
                subject=uriref, predicate=self.cum_freq_predicate, default=0
            )
            self.set(
                (uriref,
                 self.cum_freq_predicate,
                 rdflib.Literal(old_freq+cpt_freq))
            )

    def query_and_add_cpt_frequencies(self, sparql_endpoint, cpt_freq_graph,
                                      server=None, pid=None, auth_data=None):
        self.query_thesaurus(pid=pid, server=server, auth_data=auth_data)
        cpt_freqs = query_cpt_freqs(sparql_endpoint, cpt_freq_graph)
        for cpt_uri in cpt_freqs:
            cpt_atts = cpt_freqs[cpt_uri]
            cpt_freq = cpt_atts['frequency']
            if cpt_freq > 0:
                self.add_frequencies(cpt_uri, cpt_freq)

    def query_thesaurus(self, pid, server, auth_data):
        r = pp_api.export_project(
            pid=pid, server=server,
            auth_data=auth_data
        )
        self.parse(data=r, format='n3')
        top_cpts = {x[0] for x in self.triples((
            None,
            rdflib.namespace.SKOS.topConceptOf,
            None
        ))}
        for top_cpt in top_cpts:
            self.add(
                (top_cpt, rdflib.namespace.SKOS.broader, self.top_uri)
            )
            self.add(
                (self.top_uri, rdflib.namespace.SKOS.narrower, top_cpt)
            )

    def get_lcs(self, c1_uri, c2_uri):
        c1_uri = rdflib.URIRef(c1_uri)
        c2_uri = rdflib.URIRef(c2_uri)
        if c1_uri == c2_uri:
            lcs = c1_uri
            freq = self.value(subject=c1_uri,
                              predicate=self.cum_freq_predicate,
                              default=rdflib.Literal(0))
        else:
            path1 = self.triples((
                c1_uri,
                rdflib.namespace.SKOS.broader * '+',
                None
            ))
            objects_path1 = [x[2] for x in path1]
            path2 = self.triples((
                c2_uri,
                rdflib.namespace.SKOS.broader * '+',
                None
            ))
            objects_path2 = [x[2] for x in path2]
            cs = {
                x: self.value(subject=x,
                              predicate=self.cum_freq_predicate,
                              default=rdflib.Literal(0))
                for x in set(objects_path1) & set(objects_path2)
            }
            lcs, freq = min(cs.items(),
                            key=lambda x: x[1],
                            default=[None, rdflib.Literal(float('Inf'))])
        return lcs, freq.value

    def get_cumulative_freq(self, c_uri):
        c_uri = rdflib.URIRef(c_uri)
        return self.value(
            subject=c_uri,
            predicate=self.cum_freq_predicate,
            default=rdflib.Literal(0)
        ).value

    def get_own_freq(self, c_uri):
        c_uri = rdflib.URIRef(c_uri)
        return self.value(
            subject=c_uri,
            predicate=self.own_freq_predicate,
            default=rdflib.Literal(0)
        ).value

    def get_lin_similarity(self, c1_uri, c2_uri):
        if c1_uri == c2_uri:
            return 1
        c1_uri = rdflib.URIRef(c1_uri)
        c2_uri = rdflib.URIRef(c2_uri)
        lcs, lcs_freq = self.get_lcs(c1_uri, c2_uri)
        c1_freq = self.get_cumulative_freq(c1_uri)
        c2_freq = self.get_cumulative_freq(c2_uri)
        if lcs is None or c1_freq == 0 or c2_freq == 0:
            return 0
        else:
            top_freq = self.get_cumulative_freq(self.top_uri)
            p1 = np.log(c1_freq/top_freq)
            p2 = np.log(c2_freq/top_freq)
            if p1 + p2 == 0:
                return 1
            score = (2 * np.log(lcs_freq/top_freq) /
                     (p1 + p2))
            assert 0. <= score <= 1, print(lcs, score)
            return score

    def plot_layout(self):
        """
        Calculate and return positions of nodes in the taxonomy tree for 
        drawing. The labels are the prefLabels of the concepts.
        
        :return: graph, positions of nodes 
        """
        import networkx as nx
        G = nx.DiGraph()

        nodes = [self.get_pref_label(x).toPython()
                 for x in self.get_all_concepts()]
        G.add_nodes_from(nodes)
        for edge in self.triples((None, rdflib.namespace.SKOS.broader, None)):
            broader = self.get_pref_label(edge[0]).toPython()
            narrower = self.get_pref_label(edge[2]).toPython()
            G.add_edge(narrower, broader)
        pos = nx.drawing.nx_agraph.graphviz_layout(
            G, prog='twopi', args='-Goverlap=scalexy -Nroot=true -Groot=:T'
        )
        return G, pos

    def get_importance_ranking(self, method='pr'):
        """
        :param method: one of: 'pr' - PageRank, 'betweenness'. More to follow. 
        :return: dict {node: score}
        """
        import networkx as nx
        G = nx.DiGraph()

        nodes = [self.get_pref_label(x).toPython() for x in
                 self.get_all_concepts()]
        G.add_nodes_from(nodes)
        for edge in self.triples((None, rdflib.namespace.SKOS.broader, None)):
            broader = self.get_pref_label(edge[0]).toPython()
            narrower = self.get_pref_label(edge[2]).toPython()
            G.add_edge(narrower, broader)

        if method == 'betweenness':
            result = nx.betweenness_centrality(G)
        elif method == 'pr':
            result = nx.pagerank_scipy(G)
        else:
            raise Exception('Method {} not implemented yet'.format(method))
        return result

    def __iter__(self):
        all_cpts = set(self.get_all_concepts())
        for x in all_cpts:
            yield x

    def pickle(self, path):
        import dill
        with open(path, 'wb') as f:
            dill.dump(self, f)

    def __str__(self):
        out = 'Thesaurus'
        return out

    def parse_file_and_add_frequencies(self,
                                       sparql_endpoint,
                                       cpt_freq_graph,
                                       file_name,
                                       format='n3',
                                       **kwargs):

        cpt_freqs = query_cpt_freqs(sparql_endpoint, cpt_freq_graph)
        self.parse(file_name, format=format)
        for cpt_uri in cpt_freqs:
            cpt_atts = cpt_freqs[cpt_uri]
            cpt_freq = cpt_atts['frequency']
            if cpt_freq > 0:
                self.add_frequencies(cpt_uri, cpt_freq)


def query_cpt_freqs(sparql_endpoint, cpt_occur_graph):
    q_cpts = """
    select distinct ?s ?label ?freq ?cpt where {
      ?s <http://schema.semantic-web.at/ppcm/2013/5/frequencyInCorpus> ?freq .
      ?s <http://schema.semantic-web.at/ppcm/2013/5/mainLabel> ?label .
      ?s <http://schema.semantic-web.at/ppcm/2013/5/concept> ?cpt .
    }
    """
    rs = pp_api.query_sparql_endpoint(sparql_endpoint,
                                      cpt_occur_graph,
                                      q_cpts)
    results = dict()
    for r in rs:
        cpt_atts = {
            'frequency': float(r[2]),
            'mainLabel': str(r[1])
        }
        cpt_uri = r[3]
        results[cpt_uri] = cpt_atts
    return results


if __name__ == '__main__':
    import networkx as nx
    import os
    import matplotlib.pyplot as plt
    from pp_api.server_data.custom_apps import pid, server, \
        corpus_id, sparql_endpoint, pp_sparql_endpoint, p_name

    username = input('Username: ')
    pw = input('Password: ')
    auth_data = (username, pw)

    corpusgraph_id, termsgraph_id, cpt_occur_graph_id, cooc_graph = \
        pp_api.get_corpus_analysis_graphs(corpus_id)

    the = Thesaurus()
    the_path = 'misc/the.n3'
    if os.path.exists(the_path):
        the.parse(the_path, format='n3')
    else:
        the.query_thesaurus(pid=pid, server=server, auth_data=auth_data)
        the.query_and_add_cpt_frequencies(pp_sparql_endpoint + p_name,
                                          cpt_occur_graph_id,
                                          server, pid, auth_data)
        the.serialize(the_path, format='n3')

    G, pos = the.plot_layout()
    nx.draw(G, pos, with_labels=False, arrows=True, node_size=50)
    plt.title('draw_networkx')
    plt.savefig('misc/nx_test.png')

    pr = the.get_importance_ranking()
    print('PageRank: ', sorted(pr.items(),
                               key=lambda x: x[1],
                               reverse=True)[:10])

    bw = the.get_importance_ranking('betweenness')
    print('betweenness: ', sorted(bw.items(),
                                  key=lambda x: x[1],
                                  reverse=True)[:10])
