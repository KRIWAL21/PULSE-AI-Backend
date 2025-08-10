from langchain.prompts import ChatPromptTemplate

# Create the prompt template
# This tells the model how to use the context from your documents
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Provide a detailed and clear answer. If the answer is not in the context,
say, "I couldn't find the answer in the provided documents."

<context>
{context}
</context>
""")
