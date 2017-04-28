import numpy as np

import rdflib
import requests

import pp_api


class Thesaurus(rdflib.graph.Graph):
    """
    A class for thesauri. Especially useful if you have a corpus for tagging
    with this thesaurus.
    
    In order to get the cumulative concept frequencies you may want to do:
    > cpt_freqs = pp_api.get_cpt_corpus_freqs(corpus_id, server, pid, auth_data)
    >
    > the = Thesaurus()
    > for cpt in cpt_freqs:
    >     cpt_uri = cpt['concept']['uri']
    >     cpt_freq = cpt['frequency']
    >     cpt_path = pp_api.get_cpt_path(cpt_uri, server, pid, auth_data)
    >     the.add_path(cpt_path)
    >     the.add_frequencies(cpt_uri, cpt_freq)
    """

    freq_predicate = rdflib.URIRef(':frequency')

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
        return [x[0].toPython() for x in s_pl]

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
            brs = [x[2] for x in self.triples((
                uriref,
                rdflib.namespace.SKOS.broader * '*',
                None
            ))]
            path = brs
        else:
            path = [x[0] for x in path]
        for uri in path:
            uriref = rdflib.URIRef(uri)
            old_freq = self.value(
                subject=uriref, predicate=self.freq_predicate, default=0
            )
            self.set(
                (uriref, self.freq_predicate, rdflib.Literal(old_freq+cpt_freq))
            )

    def query_and_add_cpt_frequencies(self, sparql_endpoint, cpt_freq_graph,
                                      server, pid, auth_data):
        cpt_freqs = query_cpt_freqs(sparql_endpoint, cpt_freq_graph)
        s = requests.session()
        s.auth = auth_data
        for cpt_uri in cpt_freqs:
            cpt_atts = cpt_freqs[cpt_uri]
            cpt_freq = cpt_atts['frequency']
            cpt_path = pp_api.get_cpt_path(cpt_uri, server, pid, session=s)
            self.add_path(cpt_path)
            self.add_frequencies(cpt_uri, cpt_freq)

    def get_lcs(self, c1_uri, c2_uri):
        c1_uri = rdflib.URIRef(c1_uri)
        c2_uri = rdflib.URIRef(c2_uri)
        if c1_uri == c2_uri:
            lcs = c1_uri
            freq = self.value(subject=c1_uri,
                              predicate=self.freq_predicate,
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
                              predicate=self.freq_predicate,
                              default=rdflib.Literal(0))
                for x in set(objects_path1) & set(objects_path2)
            }
            lcs, freq = min(cs.items(),
                            key=lambda x: x[1],
                            default=[None, rdflib.Literal(float('Inf'))])
        return lcs, freq.value

    def get_freq(self, c_uri):
        return self.value(
            subject=c_uri,
            predicate=self.freq_predicate,
            default=rdflib.Literal(0)
        ).value

    def get_lin_similarity(self, c1_uri, c2_uri):
        if c1_uri == c2_uri:
            return 1
        c1_uri = rdflib.URIRef(c1_uri)
        c2_uri = rdflib.URIRef(c2_uri)
        lcs, lcs_freq = self.get_lcs(c1_uri, c2_uri)
        c1_freq = self.value(subject=c1_uri,
                             predicate=self.freq_predicate,
                             default=rdflib.Literal(0)).value
        c2_freq = self.value(subject=c2_uri,
                             predicate=self.freq_predicate,
                             default=rdflib.Literal(0)).value
        if lcs is None or c1_freq == 0 or c2_freq == 0:
            return 0
        else:
            top_freq = self.get_freq(self.top_uri)
            score = (2 * np.log(lcs_freq/top_freq) /
                     (np.log(c1_freq/top_freq) + np.log(c2_freq/top_freq)))
            assert 0. <= score <= 1, print(lcs, score)
            return score

    def __iter__(self):
        all_cpts = set(self.get_all_concepts())
        all_nonT_cpts = all_cpts - {self.top_uri.toPython()}
        for x in all_nonT_cpts:
            yield x

    def pickle(self, path):
        import dill
        with open(path, 'wb') as f:
            dill.dump(self, f)

    def __str__(self):
        out = 'Thesaurus'
        return out


def query_cpt_freqs(sparql_endpoint, cpt_occur_graph):
    q_cpts = """
    select distinct ?s ?label ?pl ?freq ?cpt where {
      ?s <http://schema.semantic-web.at/ppcm/2013/5/frequencyInCorpus> ?freq .
      ?s <http://schema.semantic-web.at/ppcm/2013/5/mainLabel> ?label .
      ?s <http://schema.semantic-web.at/ppcm/2013/5/concept> ?cpt .
      ?cpt <http://www.w3.org/2004/02/skos/core#prefLabel> ?pl
    }
    """
    rs = pp_api.query_sparql_endpoint(sparql_endpoint,
                                      cpt_occur_graph,
                                      q_cpts)
    results = dict()
    for r in rs:
        cpt_atts = {
            'frequency': float(r[3]),
            'prefLabel': str(r[2])
        }
        cpt_uri = r[4]
        results[cpt_uri] = cpt_atts
    return results


if __name__ == '__main__':
    pass
