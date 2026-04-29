from retrieval.google_search import GoogleSearchAPI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# class RAGPipeline:
#     def __init__(self, llm):
#         self.llm = llm
#         self.search_api = GoogleSearchAPI()

#     def build_prompt(self, query, context):
#         return f"""

# You are a Machine Learning assistant.

# Answer the question ONLY using the information from the context below.
# If the answer is not in the context, say "I don't know".

# Give a clear and short definition (1-2 sentences).

# Context:
# {context}
# Question: {query}
# Answer:
# """

#     def run(self, query):
#         # 1. Retrieve
#         results = self.search_api.search(query)
#         context = "\n".join(results)
#         print("=========================CONTEXT USED=====================")
#         print(context)
#         print("==========================================================")
#         # 2. Prompt
#         prompt = self.build_prompt(query, context)

#         # 3. Generate
#         answer = self.llm.generate(prompt)
#         return answer




class RAGPipeline:
    def __init__(self, llm):
        self.llm = llm

#     def build_prompt(self, query, context):
#         return f"""
# You are a Machine Learning assistant.
# Answer the question ONLY using the information from the context below.
# If the answer is not in the context, say "I don't know".

# Give a clear and short definition (1-2 sentences).

# Context:
# {context}

# Question: {query}
# Answer:
# """
    
        self.prompt = PromptTemplate.from_template("""
            You are a helpful assistant.

            Use ONLY the context below to answer the question.
            If the answer is not in the context, say "I don't know".

            Context:
            {context}

            Question:
            {question}

            Answer:
        """)

    def run(self, query, retriever, k=3):
        """
        Modified to accept any retriever (BM25, Dense, Google, Hybrid)
        """
        # 1. Retrieve (Works with any class following our BaseRetriever design)
        # also changed this for time
        results, retrieval_time = retriever.retrieve(query, k=k)
        
        # Check if results are objects (from our JSON) or strings (from Google)
        if isinstance(results[0], dict):
            context = "\n".join([res['text'] for res in results])
        else:
            context = "\n".join(results)

        print(f"\n=== CONTEXT USED VIA {retriever.__class__.__name__} ===")
        print(context[:500] + "...") # Print first 500 chars for brevity
        print("==========================================================\n")

        # 2. Prompt
        # prompt = self.build_prompt(query, context)
        #         # 3. Build prompt
        prompt = self.prompt.invoke({
            "context": context,
            "question": query
        })

        # 3. Generate
        answer = self.llm.generate(prompt)
        return answer
