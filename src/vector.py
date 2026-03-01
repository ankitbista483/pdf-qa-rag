from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


import tempfile
import os

class Embedder:
    def __init__(self, collection_name='QandA'):
        self.model = SentenceTransformer('BAAI/bge-m3', device='cpu')
        db_path = os.path.join(tempfile.gettempdir(), 'my_db')
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def embedder(self, chunks, metadatas):
        clean = [(c, m) for c, m in zip(chunks, metadatas) if isinstance(c, str) and c.strip()]
        chunks = [str(c).encode('utf-8', errors='ignore').decode('utf-8') for c, _ in clean]
        metadatas = [m for _, m in clean]
        for i, c in enumerate(chunks):
            if not isinstance(c, str):
                print(f"Bad chunk at {i}: {type(c)} - {c}")
        
        embedding = self.model.encode(chunks, show_progress_bar=True)
        pdf_name = metadatas[0]['Source']
        ids = [f"{pdf_name}_{i}" for i in range(len(chunks))]
        self.collection.upsert(
            documents=chunks,
            embeddings=embedding.tolist(),
            metadatas=metadatas,
            ids=ids
        )
    def is_embedded(self, filename):
        result = self.collection.get(ids=[f"{filename}_0"])
        return len(result["ids"]) > 0
    
    def get_sources(self):
        results = self.collection.get(include=['metadatas'])
        sources = list(set(m['Source'] for m in results['metadatas']))
        return sources
    
    def get_answer(self, query_text,selected_pdf= None,session_pdfs = None):
        embedded_query = self.model.encode(query_text)
        if selected_pdf and selected_pdf != "ALL PDFs":
             return self.collection.query(
                query_embeddings=[embedded_query.tolist()],
                n_results=3,
                where= {"Source" : selected_pdf}
            )
        elif session_pdfs:
            return self.collection.query(
                query_embeddings=[embedded_query.tolist()],
                n_results= 3,
                where= {"Source" : {"$in" : session_pdfs}}
            )
        else:
            return self.collection.query(
                query_embeddings=[embedded_query.tolist()],
                n_results=3
            )
        
    def bm25_search(self,chunks,query):
        tokenized_chunks = [chunk.split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)  
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        top_chunks = [chunks[i] for i in top_indices]
        return top_chunks
    
    def re_ranking(self,chunks,query,top_n=3):
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranks = model.rank(query,chunks)
        top_rank = [chunks[rank["corpus_id"]] for rank in ranks[:top_n]]
        return top_rank




