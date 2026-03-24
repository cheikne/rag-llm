from retrieval.google_search import GoogleSearchAPI

class RAGPipeline:
    def __init__(self, llm):
        self.llm = llm
        self.search_api = GoogleSearchAPI()

    def build_prompt(self, query, context):
        return f"""

You are a Machine Learning assistant.

Answer the question ONLY using the information from the context below.
If the answer is not in the context, say "I don't know".

Give a clear and short definition (1-2 sentences).

Context:
{context}
Question: {query}
Answer:
"""

    def run(self, query):
        # 1. Retrieve
        results = self.search_api.search(query)
        context = "\n".join(results)
        print("=========================CONTEXT USED=====================")
        print(context)
        print("==========================================================")
        # 2. Prompt
        prompt = self.build_prompt(query, context)

        # 3. Generate
        answer = self.llm.generate(prompt)
        return answer
    
    # You are a Machine Learning assistant.

# Use the context below to answer the question.

# Context:
# {context}
