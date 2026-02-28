import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from langchain_ollama import OllamaLLM
from src.text_extractor  import ExtractText
from src.vector import Embedder

file = ExtractText('978-3-031-38747-0.pdf')
file.extract_text()

embedder = Embedder()
filename = os.path.basename('978-3-031-38747-0.pdf')
if not embedder.is_embedded(filename):
    embedder.embedder(chunks=file.chunks, metadatas=file.metadatas)
else:
    print(f"{filename} already embedded, skipping...")

query = input("Ask a question: ")
results = embedder.get_answer(query)
print("\n--- RETRIEVED CHUNKS ---")
for i, doc in enumerate(results["documents"][0]):
    print(f"\nChunk {i+1}:")
    print(doc)
    print(f"Page: {results['metadatas'][0][i]}")
llm = OllamaLLM(model='llama3.1:8b')
context_text = "\n\n---\n\n".join(results["documents"][0])
prompt = f"""
You are a technical assistant. Answer the question using ONLY the context below.
If multiple chunks contain conflicting information, use the chunk that most directly and specifically answers the question.
If the answer is truly not in the context, say "I don't know".
Be concise and direct."

CONTEXT:
{context_text}

QUESTION: 
{query}

ANSWER:
"""
response = llm.invoke(prompt)
print(f"\n--- AI RESPONSE ---\n{response}")


