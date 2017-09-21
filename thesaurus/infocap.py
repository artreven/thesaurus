import rdflib
from scipy import stats

import pp_api

swc_onto = rdflib.Namespace('http://schema.semantic-web.at/ppcm/2013/5/')


def get_term_ridf(sparql_endpoint, named_graph, term):
    graph = rdflib.Graph('SPARQLStore', identifier=named_graph)
    graph.open(sparql_endpoint)

    term_uri = list(graph.triples((
        None,
        swc_onto.textValue,
        rdflib.Literal(term, datatype=rdflib.XSD.string)
    )))
    try:
        term_uri = term_uri[0][0]
        term_ridf = float(graph.value(
            term_uri,
            swc_onto.combinedRelevanceScore
        ))
    except IndexError:
        term_ridf = None
    return term_ridf


def get_doc_scores(doc_text, ridf_scores, pid, server, auth_data):
    r = pp_api.extract(doc_text, pid, server, auth_data)
    terms = pp_api.get_terms_from_response(r)
    cpts = pp_api.get_cpts_from_response(r)
    sum_ridf = 0
    for term in terms:
        term_text = term['textValue']
        try:
            term_ridf = ridf_scores[term_text]
            sum_ridf += term_ridf
        except KeyError:
            continue
    sum_cpts = sum([cpt['frequencyInDocument'] for cpt in cpts])
    return sum_ridf, sum_cpts


def get_corpus_infocap_score(sparql_endpoint, termsgraph_id, corpus_id,
                             pid, server, auth_data):
    ridfs_scores = pp_api.get_ridfs(sparql_endpoint, termsgraph_id)
    rs = pp_api.get_corpus_documents(corpus_id, pid, server, auth_data)
    term_scores = []
    cpt_scores = []
    for row in rs:
        doc_content = row['content']
        term_score, cpt_score = get_doc_scores(
            doc_content, ridfs_scores, pid, server, auth_data
        )
        term_scores.append(term_score)
        cpt_scores.append(cpt_score)
    slope, intercept, r_value, p_value, std_err = stats.linregress(term_scores,
                                                                   cpt_scores)
    return slope, intercept, r_value, p_value, std_err, term_scores, cpt_scores
