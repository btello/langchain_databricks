# Databricks notebook source
# MAGIC %md
# MAGIC # Chains in Langchain

# COMMAND ----------

# DBTITLE 1,Load test data

reviews_dataframe = (spark.sql("SELECT title as product, review FROM parquet.`dbfs:/databricks-datasets/amazon/test4K/` limit 500"))
display(reviews_dataframe)

# COMMAND ----------

from langchain.document_loaders import PySparkDataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
 
loader = PySparkDataFrameLoader(spark, reviews_dataframe, page_content_column="review")
documents = loader.load()
 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(f"Number of documents: {len(texts)}")

# COMMAND ----------

# DBTITLE 1,Load LLM
from langchain.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatDatabricks(
    target_uri="databricks",
    endpoint="databricks-llama-2-70b-chat",
    temperature=0.1,
)

# COMMAND ----------

prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}? Simply output the name you suggest with no other text"
)

chain = LLMChain(llm=llm, prompt=prompt)
product = "Queen Size Sheet Set"
chain.run(product)

# COMMAND ----------

# DBTITLE 1,Sequential Chains
from langchain.chains import SimpleSequentialChain

# COMMAND ----------

# MAGIC %md
# MAGIC Simple Sequential Chain is for chains that expect one input and return one output

# COMMAND ----------

llm = ChatDatabricks(
    target_uri="databricks",
    endpoint="databricks-llama-2-70b-chat",
    temperature=0.1,
)

# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# COMMAND ----------

# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)

# COMMAND ----------

overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )

# COMMAND ----------

overall_simple_chain.run(product)

# COMMAND ----------

# DBTITLE 1,Sequential Chains
# MAGIC %md
# MAGIC
# MAGIC Sequential Chains are useful for the case where you have more than one input and output

# COMMAND ----------

from langchain.chains import SequentialChain

# COMMAND ----------

llm = ChatDatabricks(
    target_uri="databricks",
    endpoint="databricks-llama-2-70b-chat",
    temperature=0.1,
)

# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="English_Review"
                    )

# COMMAND ----------

second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
# chain 2: input= English_Review and output= summary
chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="summary"
                    )

# COMMAND ----------

# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
# chain 3: input= Review and output= language
chain_three = LLMChain(llm=llm, prompt=third_prompt,
                       output_key="language"
                      )

# COMMAND ----------

# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     )

# COMMAND ----------

# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
)

# COMMAND ----------

# DBTITLE 1,Get a random review to pass to the chain
# Import necessary libraries
from pyspark.sql.functions import col
import random

# Select a random review from the DataFrame
reviews_count = reviews_dataframe.count()
random_index = random.randint(0, reviews_count - 1)
random_review = reviews_dataframe.select(col("review")).take(random_index + 1)[-1][0]

# Store the random review as a Python string in the 'review' variable
review = str(random_review)
print(review)

# COMMAND ----------

# DBTITLE 1,Translate review to another language to see how the chain works!
from langchain.prompts import ChatPromptTemplate

template_string = """Translate the text \
that is delimited by triple backticks \
into {language}.  Do not include any other commentary, simply output the translation. \
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

customer_messages = prompt_template.format_messages(
                    language="Spanish",
                    text=review)

response = llm(customer_messages)

translated_review = response.content
print(translated_review)

# COMMAND ----------

overall_chain(translated_review)

# COMMAND ----------

# DBTITLE 1,Router Chains
# MAGIC %md 
# MAGIC
# MAGIC Sometimes you may want to run a chain based on some sort of criteria (are the user asking about computer science?  Physics? 
# MAGIC  etc.).  This is where router chains come in handy!

# COMMAND ----------

# DBTITLE 1,Router Chains
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""


math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""


computerscience_template = """ You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity. 

Here is a question:
{input}"""

# COMMAND ----------

prompt_infos = [
    {
        "name": "physics", 
        "description": "Good for answering questions about physics", 
        "prompt_template": physics_template
    },
    {
        "name": "math", 
        "description": "Good for answering math questions", 
        "prompt_template": math_template
    },
    {
        "name": "History", 
        "description": "Good for answering history questions", 
        "prompt_template": history_template
    },
    {
        "name": "computer science", 
        "description": "Good for answering computer science questions", 
        "prompt_template": computerscience_template
    }
]

# COMMAND ----------

from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate

# COMMAND ----------

llm = ChatDatabricks(
    target_uri="databricks",
    endpoint="databricks-llama-2-70b-chat",
    temperature=0.0,
)

# COMMAND ----------

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain  
    
destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
print(destinations_str)

# COMMAND ----------

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llm, prompt=default_prompt)

# COMMAND ----------

MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

Only output the json as directed an no further commentary.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

# COMMAND ----------

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# COMMAND ----------

chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        )

# COMMAND ----------

chain.run("What is black body radiation?")

# COMMAND ----------

chain.run("What is the fastest algorithm for finding an item in a sorted list")

# COMMAND ----------


