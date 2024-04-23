# Databricks notebook source
# MAGIC %md
# MAGIC # Prompt Templates

# COMMAND ----------

# DBTITLE 1,Setting up langchain against a foundation model
from langchain.chat_models import ChatDatabricks
from langchain_core.messages import HumanMessage

chat = ChatDatabricks(
    target_uri="databricks",
    endpoint="databricks-llama-2-70b-chat",
    temperature=0.1,
)

# COMMAND ----------

# DBTITLE 1,Setting up a prompt template
from langchain.prompts import ChatPromptTemplate

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

customer_style = """American English \
in a calm and respectful tone
"""

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse, \
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

customer_messages = prompt_template.format_messages(
                    style=customer_style,
                    text=customer_email)

# Call the LLM to translate to the style of the customer message
customer_response = chat(customer_messages)

print(customer_response.content)

# COMMAND ----------

# DBTITLE 1,Re-using the prompt template for reverse translation
service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_pirate = """\
a polite tone \
that speaks in English Pirate\
"""

service_messages = prompt_template.format_messages(
    style=service_style_pirate,
    text=service_reply)

service_response = chat(service_messages)

print(service_response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Output Parsers

# COMMAND ----------

# MAGIC %md
# MAGIC Let's start with defining what we would like the LLM output to look like.

# COMMAND ----------

{
  "gift": False,
  "delivery_days": 5,
  "price_value": "pretty affordable!"
}

# COMMAND ----------

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

Do not provide any additional reasoning, simply output the json as described

text: {text}
"""

# COMMAND ----------

from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(review_template)
print(prompt_template)

# COMMAND ----------

messages = prompt_template.format_messages(text=customer_review)
response = chat(messages)
print(response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC The type of `response.content` is a string but what if we actually wanted to get the value of one of those fields, say `gift`?  That's where output parsers come in to play.

# COMMAND ----------

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer true if yes,\
                             false if not or unknown.")
delivery_days_schema = ResponseSchema(name="delivery_days",
                                      description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",
                                    description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")

response_schemas = [gift_schema, 
                    delivery_days_schema,
                    price_value_schema]

# COMMAND ----------

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# COMMAND ----------

format_instructions = output_parser.get_format_instructions()

# COMMAND ----------

print(format_instructions)

# COMMAND ----------

review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer true if yes, false if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}

Do not supply any additional reasoning, simply print the JSON as instructed
"""

prompt = ChatPromptTemplate.from_template(template=review_template_2)

messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instructions)

# COMMAND ----------

chat = ChatDatabricks(
    target_uri="databricks",
    endpoint="databricks-llama-2-70b-chat",
    temperature=1,
)

response = chat(messages)

# COMMAND ----------

print(response.content)

# COMMAND ----------

output_dict = output_parser.parse(response.content)

# COMMAND ----------

print(output_dict)

# COMMAND ----------

print(output_dict['gift'])

# COMMAND ----------

print(output_dict['delivery_days'])

# COMMAND ----------


