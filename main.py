from model.llm import LLMModel
from rag.pipeline_rag import RAGPipeline
def clean_output(text):
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()
def main():
    llm = LLMModel()
    rag = RAGPipeline(llm)

    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        answer = rag.run(query)
        print("=========================ANSWER=====================")
        print(clean_output(answer))

if __name__ == "__main__":
    main()