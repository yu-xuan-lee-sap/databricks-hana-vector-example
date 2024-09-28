# Example of Vector Indexing Pipeline with Databricks, HANA Vector DB and Generative AI Hub SDK

## About Repo

This repo shows an example of how a vector indexing pipeline for Retrieval-Augmented Generation (RAG) can be created using Databricks, HANA Vector DB and Generative AI Hub SDK.

This repo shows pipeline codes in two different format, both of which work in Databricks:
- A [Python source file](./notebooks/etl_source.py) containing the codes of the Extract, Load, Transform steps of the indexing pipeline
- A [Jupyter Notebook](./notebooks/etl_jupyter.ipynb) containing the same codes as the Python source file

## Reproducing this Repo

### Prerequisites
- Access to Databricks
- Databricks cluster runtime: 15.4 LTS ML or above
- Secrets for HANA DB [[example](https://github.com/yu-xuan-lee-sap/rag-langchain-hana-example/blob/main/secrets/hana-secrets-example.json)]
- Secrets for Generative AI Hub [[example](https://github.com/yu-xuan-lee-sap/rag-langchain-hana-example/blob/main/secrets/gen-ai-hub-service-key-example.json)]

### Cloning Repo and Running Codes
Refer to Databricks documentation on
- [How to clone repos](https://docs.databricks.com/en/repos/git-operations-with-repos.html)
- [How to work with Databricks notebooks](https://docs.databricks.com/en/notebooks/index.html)