

from langchain_ollama.llms import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate

from vector import retriever




#  Initialize the local language model
model = OllamaLLM(model="llama3.2")


# 3. Create the prompt template
template = """
You are an expert in answering questions about restaurants.

Use only the following restaurant reviews to answer the question.
If the reviews do not contain enough information, say so honestly.

Relevant reviews:
{reviews}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)


# 4. Connect the prompt and model
chain = prompt | model


while True:
  print("\n\n--------------------")
  question= input("ask your question( press q to quit):")
  print("\n\n")
  if question.lower() == "q":
    break

  reviews = retriever.invoke(question)
  result = chain.invoke({
      "reviews": reviews,
      "question": question
  })


  print("\nAnswer:", result)
