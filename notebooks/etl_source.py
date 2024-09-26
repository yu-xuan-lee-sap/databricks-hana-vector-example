# Databricks notebook source
# MAGIC %md
# MAGIC # Example of Databricks Job Notebook for Vector Indexing using HANA Vector DB and Generative AI Hub SDK
# MAGIC
# MAGIC This notebook shows an example of how HANA Vector DB and Generative AI Hub SDK can be used in a Databricks Job to generate embeddings from data in Azure Data Lake Service (ADLS), and writing these embedding vectors to HANA Vector DB. The entire process is split into the Extract, Transform, Load phases in this notebook, as this is essentially a data engineering process.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# MAGIC %md
# MAGIC Install required packages
# MAGIC - `html2text`: Converting text from HTML to Markdown format
# MAGIC - `generative-ai-hub-sdk`: For working with Generative AI models in GenAI Hub
# MAGIC - `hdbcli`: For connection to HANA Vector DB

# COMMAND ----------

# MAGIC %pip install html2text "generative-ai-hub-sdk[all]" hdbcli
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract

# COMMAND ----------

# MAGIC %md
# MAGIC Ingest raw HTML files from ADLS

# COMMAND ----------

from pyspark.sql.functions import expr

HTML_FILEPATHS_GLOB_PATTERN = "abfss://aitm@coredatalaketestint.dfs.core.windows.net/sandbox/example_wikis/*"

html_binary_spark_df = (
    spark
    .read
    .format("binaryFile")
    .options(pathGlobFilter="*.html")
    .load(HTML_FILEPATHS_GLOB_PATTERN)
    .withColumn("content", expr("CAST(content AS STRING)"))
)

# COMMAND ----------

display(html_binary_spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform

# COMMAND ----------

# MAGIC %md
# MAGIC Convert wiki texts from HTML to Markdown, and clean the text contents

# COMMAND ----------

import re
import html2text
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType

AUTHORS_REGEX_PATTERN = re.compile(r"Created\sby.*last\smodified.*[A-Z][a-z]{2}\s\d{2}\,\s\d{4}")

convert_html_to_markdown = udf(html2text.html2text, StringType())

@udf(StringType())
def strip_header_and_footer(content: str) -> str:
    return content[53:-100]

@udf(StringType())
def remove_authors(content: str) -> str:
    return AUTHORS_REGEX_PATTERN.sub("", content)

@udf(StringType())
def get_filename(path: str) -> str:
    return path.split("/")[-1]

wiki_texts_spark_df = (
    html_binary_spark_df
    .withColumn("content", convert_html_to_markdown(col("content")))
    .withColumn("content", strip_header_and_footer(col("content")))
    .withColumn("content", remove_authors(col("content")))
    .withColumn("filename", get_filename(col("path")))
    .selectExpr("filename", "content")
)

# COMMAND ----------

display(wiki_texts_spark_df.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC Split the wiki texts into chunks

# COMMAND ----------

from langchain_community.document_loaders import PySparkDataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

full_contents_loader = PySparkDataFrameLoader(spark_session=spark, df=wiki_texts_spark_df, page_content_column="content")
documents_to_split = full_contents_loader.load()

document_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=200,
    chunk_overlap=20
)

document_chunks = document_splitter.split_documents(documents_to_split)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load

# COMMAND ----------

# MAGIC %md
# MAGIC Read secrets for GenAI Hub and HANA Vector DB from secret scope

# COMMAND ----------

import json

SECRET_SCOPE = "TEST_AITM_SCOPE"
gen_ai_hub_service_key = json.loads(dbutils.secrets.get(scope=SECRET_SCOPE, key="EXAMPLE_GENAI_HUB_SERVICE_KEY"))
hana_secrets = json.loads(dbutils.secrets.get(scope=SECRET_SCOPE, key="EXAMPLE_HANA_VECTOR_SECRETS"))

# COMMAND ----------

# MAGIC %md
# MAGIC Initialize connection object for HANA Vector DB

# COMMAND ----------

from hdbcli import dbapi

hana_conn = dbapi.connect(
    address=hana_secrets["host"],
    port=hana_secrets["port"],
    user=hana_secrets["user"],
    password=hana_secrets["password"],
    autocommit=True,
    sslTrustStore=hana_secrets["certificate"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC Initialize client for GenAI Hub

# COMMAND ----------

import os
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

os.environ["AICORE_AUTH_URL"] = gen_ai_hub_service_key["url"]
os.environ["AICORE_CLIENT_ID"] = gen_ai_hub_service_key["clientid"]
os.environ["AICORE_CLIENT_SECRET"] = gen_ai_hub_service_key["clientsecret"]
os.environ["AICORE_RESOURCE_GROUP"] = gen_ai_hub_service_key["appname"].split("!")[0]
os.environ["AICORE_BASE_URL"] = f"{gen_ai_hub_service_key['serviceurls']['AI_API_URL']}/v2"

proxy_client = get_proxy_client("gen-ai-hub")

# COMMAND ----------

# MAGIC %md
# MAGIC Define LangChain object for embedding model, and initialize LangChain object for working with HANA Vector DB

# COMMAND ----------

from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from langchain_community.vectorstores import HanaDB

embeddings = init_embedding_model("text-embedding-ada-002", proxy_client=proxy_client)
hana_vectordb = HanaDB(embedding=embeddings, connection=hana_conn, table_name="DATABRICKS_HANA_EXAMPLE_VECTORSTORE")

# COMMAND ----------

# MAGIC %md
# MAGIC Write chunks to HANA Vector DB

# COMMAND ----------

hana_vectordb.add_documents(document_chunks)

# COMMAND ----------

# MAGIC %md
# MAGIC \[Optional\] Send query via in-built retriever object for retrieval of context

# COMMAND ----------

hana_vector_retriever = hana_vectordb.as_retriever()

hana_vector_retriever.get_relevant_documents("What is RAG?")

# COMMAND ----------


