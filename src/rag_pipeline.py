# Dual‑mode Spaces / ZeroGPU support
import os
HF_DEPLOYMENT = os.getenv("HF_DEPLOYMENT", "false").lower() == "true"

if HF_DEPLOYMENT:
    import spaces
else:
    class FakeSpaces:
        def GPU(self, fn):
            # No‑op decorator for local testing
            return fn
    spaces = FakeSpaces()

from typing import List, Tuple, Dict, Any, Union
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_groq import ChatGroq
from langchain_core.documents import Document

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from src.config import (
    DOCS_DIR,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LOCAL_LLM_MODEL_NAME,
    TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    URLS
)

from dotenv import load_dotenv
load_dotenv()

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline:
    - Load documents (PDF, MD, TXT, URL)
    - Build FAISS index with SentenceTransformer embeddings
    - Retrieve top-k chunks
    - Call Groq LLM via LangChain OR local fallback LLM
    - Return answer + source attribution
    """
    
    def __init__(self):
        # Embeddings + text splitter
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        self.documents: List[Document] = []
        self.index = None
        self.doc_embeddings = None

        # LLM: Groq if available, else local Qwen
        if 'GROQ_API_KEY' in os.environ:
            self.llm: Union[ChatGroq, Tuple[Any, Any]] = ChatGroq(
                api_key=GROQ_API_KEY,
                model=GROQ_MODEL_NAME,
                temperature=0.2,
                max_retries=2
            )
            self.use_groq = True
            print("Using Groq LLM")
        else:
            self.llm = self._load_local_llm()
            self.use_groq = False
            print("Using local Qwen LLM")

        # Build index once at startup
        self._load_and_index_documents()

    # Keep ZeroGPU happy by triggering the GPU layer while letting the main loop run on CPU.
    @spaces.GPU(duration=1)
    def dummy_gpu_startup_trigger():
        """ This function triggers initially the GPU layer the Hugging Face startup scanner. """
        return "ZeroGPU Layer Initialized Successfully"

    def _load_local_llm(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load local fallback LLM (Qwen2.5-0.5B-Instruct) from Hugging Face.
        In local mode, this still runs but without Spaces GPU orchestration.
        """
        model_name = LOCAL_LLM_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        return model, tokenizer

    def _local_llm_generate(self, prompt: str) -> str:
        """
        Generate text using local fallback LLM.
        """
        model, tokenizer = self.llm
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _load_local_docs(self) -> List[Document]:
        """
        Load local documents from data/docs/ (PDF, MD, TXT).
        Adds source metadata for attribution.
        """
        docs: List[Document] = []
        docs_path = Path(DOCS_DIR)

        if not docs_path.exists():
            return docs

        for file in docs_path.iterdir():
            if file.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file))
            elif file.suffix.lower() == ".md":
                loader = UnstructuredMarkdownLoader(str(file))
            elif file.suffix.lower() == ".txt":
                loader = TextLoader(str(file))
            else:
                continue

            loaded = loader.load()
            for d in loaded:
                d.metadata["source"] = str(file)
                d.metadata["type"] = file.suffix.lower()
            docs.extend(loaded)

        return docs

    def _load_url_docs(self) -> List[Document]:
        """
        Optionally load URL-based documents.
        Define URLs here or extend via config.
        """
        urls = URLS
        if not urls:
            return []

        loader = WebBaseLoader(urls)
        loaded = loader.load()
        for d, url in zip(loaded, urls):
            d.metadata["source"] = url
            d.metadata["type"] = "url"
        return loaded

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts using SentenceTransformer.
        """
        return self.embedding_model.encode(texts, convert_to_numpy=True)

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        """
        return self.embedding_model.encode([query], convert_to_numpy=True)

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """
        Build a FAISS index from embeddings.
        """
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def _load_and_index_documents(self):
        """
        Load documents, split into chunks, embed, and build FAISS index.
        Runs once at startup (app.py).
        """
        local_docs = self._load_local_docs()
        url_docs = self._load_url_docs()
        all_docs = local_docs + url_docs

        if not all_docs:
            self.documents = []
            self.index = None
            self.doc_embeddings = None
            return

        split_docs = self.text_splitter.split_documents(all_docs)
        self.documents = split_docs

        texts = [doc.page_content for doc in split_docs]
        embeddings = self._embed_texts(texts)

        index = self._build_faiss_index(embeddings)

        self.index = index
        self.doc_embeddings = embeddings

        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        faiss.write_index(index, str(FAISS_INDEX_PATH))

    def _search(self, query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
        """
        Search FAISS index for top-k similar chunks.

        Returns:
            List of (Document, distance).
        """
        if self.index is None or self.doc_embeddings is None or not self.documents:
            return []

        query_emb = self._embed_query(query)
        distances, indices = self.index.search(query_emb, k)

        results: List[Tuple[Document, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append((doc, float(dist)))
        return results

    def _call_llm(self, prompt: str) -> str:
        """
        Unified LLM call: Groq if available, else local Qwen.
        Decorated for ZeroGPU in HF Spaces; no-op locally.
        """
        if self.use_groq:
            response = self.llm.invoke(prompt)
            return response.content
        else:
            return self._local_llm_generate(prompt)

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Run RAG: retrieve relevant chunks and generate an answer via LLM.

        Returns:
            Dict with 'answer' and 'sources' (list of source strings).
        """
        results = self._search(user_query, k=TOP_K)

        if not results:
            prompt = (
                "You are a cautious medical assistant chatbot focused on heart disease.\n"
                "No external context documents are available.\n"
                f"User question:\n{user_query}\n\n"
                "Provide a general, high-level answer, avoid specific medical advice, "
                "and encourage consulting a healthcare professional."
            )
            answer = self._call_llm(prompt)
            return {
                "answer": answer,
                "sources": [],
            }

        context_chunks = []
        sources = []
        for doc, _ in results:
            src = doc.metadata.get("source", "unknown")
            context_chunks.append(f"Source ({src}):\n{doc.page_content}")
            sources.append(src)

        context = "\n\n".join(context_chunks)

        prompt = (
            "You are a medical assistant chatbot focused on heart disease.\n"
            "Use the following context to answer the user's question accurately and cautiously.\n\n"
            f"Context:\n{context}\n\n"
            f"User question:\n{user_query}\n\n"
            "Answer clearly, avoid giving direct treatment instructions, and encourage consulting a clinician."
        )

        answer = self._call_llm(prompt)
        return {
            "answer": answer,
            "sources": list(dict.fromkeys(sources)),  # dedupe while preserving order
        }
