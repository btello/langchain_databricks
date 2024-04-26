# Databricks notebook source
# MAGIC %pip install --upgrade langchain sqlalchemy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

reviews_dataframe = (spark.sql("SELECT title, brand, review FROM parquet.`dbfs:/databricks-datasets/amazon/test4K/` LIMIT 1000"))
display(reviews_dataframe)

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatDatabricks
from langchain.document_loaders import PySparkDataFrameLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import Databricks

# COMMAND ----------

loader = PySparkDataFrameLoader(spark, reviews_dataframe, page_content_column="review")
documents = loader.load()

# COMMAND ----------

from langchain.indexes import VectorstoreIndexCreator

# COMMAND ----------

# MAGIC %pip install docarray

# COMMAND ----------

# DBTITLE 1,Create an in memory index using DBRX embeddings
from langchain.embeddings import DatabricksEmbeddings

embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

index = VectorstoreIndexCreator(
    embedding = embeddings,
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# COMMAND ----------

print(embeddings.embed_query("what are you doing?"))

# COMMAND ----------

query ="which products have bad reviews?"

# COMMAND ----------

from langchain_community.llms import Databricks

def transform_input(**request):

    return { 
        "messages": [
            {
                "role": "user",
                "content": request["prompt"]
            }
        ],
        "max_tokens": 128
    }


llm_replacement_model = Databricks(endpoint_name="databricks-dbrx-instruct", transform_input_fn=transform_input)

response = index.query(query, 
                       llm = llm_replacement_model, verbose=True)

# COMMAND ----------

print(response)

# COMMAND ----------

# DBTITLE 1,Diving into in memory index
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

db = DocArrayInMemorySearch.from_documents(texts, embeddings)

# COMMAND ----------

query = "what do people think about fitbit flex?"

# COMMAND ----------

docs = db.similarity_search(query)

# COMMAND ----------

for doc in docs:
  print(doc.page_content)
  print('')

# COMMAND ----------


