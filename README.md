Thesaurus ([SKOS](https://www.w3.org/2004/02/skos/) compatible)
===============================

Defines a class for storing thesaurus and defining distances between different concepts.
Currently Lin distances are defined.
Meant to be used together with [pp_api](https://github.com/artreven/pp_api) package.

Example of usage:
```python
cpt_freqs = pp_api.get_cpt_corpus_freqs(corpus_id, server, pid, auth_data)

the = Thesaurus()
for cpt in cpt_freqs:
    cpt_uri = cpt['concept']['uri']
    cpt_freq = cpt['frequency']
    cpt_path = pp_api.get_cpt_path(cpt_uri, server, pid, auth_data)
    the.add_path(cpt_path)
    the.add_frequencies(cpt_uri, cpt_freq)
```