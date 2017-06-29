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

