## Evaluation results

### Baselin models (Table 2)

- Base models

- Feature Abstract

```python /src/hybrid/feature_simil/abstract_simil.py```

- Feature Formulae

We utlize Pya0 https://github.com/approach0/pya0 for indexing all formulae from zbMATHOpen

The documents are obtained via zbMATHOpen API https://api.zbmath.org/v1/

The ID of documents can be obtained from repo: https://zenodo.org/records/5062959

Then run

```python /src/hybrid/feature_simil/mabowdor.py```

- Feature Refereces

```python /src/hybrid/feature_simil/co_citation.py```