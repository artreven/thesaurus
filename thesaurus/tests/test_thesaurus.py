import rdflib
import os

from pp_api.server_data.custom_apps import pid, server, \
        corpus_id, sparql_endpoint, pp_sparql_endpoint, p_name

from thesaurus.thesaurus import *


class TestTopicMatrices:
    def setUp(self):
        the = Thesaurus()
        the_path = '../misc/the.n3'
        if os.path.exists(the_path):
            the.parse(the_path, format='n3')
        else:
            raise Exception('Thesaurus not found at {}'.format(
                os.path.abspath(the_path)
            ))

        self.the = the

    def tearDown(self):
        pass

    def test_leaves(self):
        leaves_the = self.the.get_leaves()
        brs = {x[2] for x in self.the.triples((
            None,
            rdflib.namespace.SKOS.broader,
            None
        ))}
        leaves_q = set(self.the.get_all_concepts()) - brs
        assert leaves_q == leaves_the

    def test_freqs(self):
        assert all(
            self.the.get_cumulative_freq(cpt) >= self.the.get_own_freq(cpt) >= 0
            for cpt in self.the.get_all_concepts()
        )
        top = self.the.top_uri
        own_freq = self.the.get_own_freq(top)
        cum_freq = self.the.get_cumulative_freq(top)
        assert own_freq == 0
        assert cum_freq > 0
        assert any(
            self.the.get_cumulative_freq(cpt) > self.the.get_own_freq(cpt)
            for cpt in self.the.get_all_concepts()
        )
        for cpt in self.the.get_leaves():
            own_freq = self.the.get_own_freq(cpt)
            cum_freq = self.the.get_cumulative_freq(cpt)
            assert cum_freq == own_freq
