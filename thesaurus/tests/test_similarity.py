import numpy as np
import pprint
import logging
from time import time
import os

from scipy import sparse

from thesaurus.thesaurus import Thesaurus, get_sim_dict
# import similarity_pipelines.pipelines as pl
import pp_api
import pp_api.server_data.custom_apps as custom

logging.basicConfig(format='%(name)s at %(asctime)s: %(message)s')
log_level_str = os.environ.get('LOG_LEVEL', logging.WARNING)
log_level = getattr(logging, str(log_level_str), logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

username = os.environ.get('pp_user')  # input("Username: ")
pw = os.environ.get('pp_password')  # input("Password: ")
auth_data = (username, pw)


class TestSimilarity():
    def setUp(self):
        self.config = {
            'server': custom.server,
            'auth_data': auth_data,
            'sparql_endpoint': custom.sparql_endpoint,
        }
        self.sim_dict_path = 'sim_dict'

    # def tearDown(self):
    #     os.remove(self.sim_dict_path)

    def tet_init_similarities_reegle(self):
        start = time()
        the = Thesaurus.get_the(
            the_path='the_no_freqs_reegle.n3', pid=custom.reegle_pid,
            server=custom.server,
            auth_data=auth_data, with_freqs=False
        )
        self.sim_dict = get_sim_dict(sim_dict_path=self.sim_dict_path,
                                     the=the)
        assert time() - start < 1000

    def test_init_similarities_eurovoc_rich(self):
        the = Thesaurus.get_the(
            the_path='the_no_freqs_ee.n3', pid=custom.ee_pid,
            server=custom.server,
            auth_data=auth_data, with_freqs=False
        )
        self.sim_dict = get_sim_dict(sim_dict_path=self.sim_dict_path,
                                     the=the)
