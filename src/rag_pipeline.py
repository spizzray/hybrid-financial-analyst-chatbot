"""
RAG Pipeline - Retrieval Augmented Generation

This module handles:
1. Loading and processing documents (PDFs and text files)
2. Splitting documents into chunks
3. Creating embeddings (numerical representations of text)
4. Storing embeddings in ChromaDB (vector database)
5. Retrieving relevant chunks when user asks questions

Think of this as building a smart search engine for your documents!
"""

import os
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


class RAGPipeline:
    """Manages document processing and retrieval for RAG"""

    def __init__(self, docs_directory: str, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG pipeline

        Args:
            docs_directory: Path to folder containing PDFs and text files
            persist_directory: Where to save the vector database
        """
        self.docs_directory = docs_directory
        self.persist_directory = persist_directory

        # Initialize the embedding model
        # This model converts text to 384-dimensional vectors
        # It runs LOCALLY (no API calls needed!)
        print("🔄 Loading embedding model (this may take a minute on first run)...")
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("🔄 Trying alternative model name...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        print("✅ Embedding model loaded")

        # Initialize ChromaDB client
        # persistent_client means the database is saved to disk
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

        # Create or get a collection (like a table in SQL)
        # A collection stores all our document embeddings
        self.collection = self.chroma_client.get_or_create_collection(
            name="financial_documents",
            metadata={"description": "Financial reports and transcripts"}
        )

        print(f"📊 ChromaDB collection has {self.collection.count()} documents")

    def load_documents(self) -> List[LangChainDocument]:
        """
        Load all PDF and text files from the documents directory

        Returns:
            List of LangChain Document objects (text + metadata)
        """
        documents = []
        docs_path = Path(self.docs_directory)

        print(f"\n📂 Loading documents from {self.docs_directory}")

        # Process each file in the directory
        for file_path in docs_path.iterdir():
            if file_path.suffix == '.pdf':
                # Handle PDF files
                docs = self._load_pdf(file_path)
                documents.extend(docs)
                print(f"  ✅ Loaded {len(docs)} pages from {file_path.name}")

            elif file_path.suffix == '.txt':
                # Handle text files
                doc = self._load_text(file_path)
                documents.append(doc)
                print(f"  ✅ Loaded {file_path.name}")

        print(f"\n📚 Total documents loaded: {len(documents)}")
        return documents

    def _load_pdf(self, file_path: Path) -> List[LangChainDocument]:
        """
        Load a PDF file and extract text from each page

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document objects (one per page)
        """
        documents = []

        try:
            # PdfReader from pypdf library
            reader = PdfReader(str(file_path))

            # Extract text from each page
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()

                # Skip empty pages
                if text.strip():
                    # Create a Document with metadata
                    doc = LangChainDocument(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "page": page_num,
                            "type": "pdf"
                        }
                    )
                    documents.append(doc)

        except Exception as e:
            print(f"  ❌ Error loading {file_path.name}: {e}")

        return documents

    def _load_text(self, file_path: Path) -> LangChainDocument:
        """
        Load a text file

        Args:
            file_path: Path to text file

        Returns:
            Document object containing the text
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        return LangChainDocument(
            page_content=text,
            metadata={
                "source": file_path.name,
                "type": "text"
            }
        )

    def split_documents(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """
        Split large documents into smaller chunks

        Why? LLMs have token limits, and smaller chunks give more precise retrieval.

        Args:
            documents: List of Document objects

        Returns:
            List of Document chunks
        """
        # RecursiveCharacterTextSplitter intelligently splits text
        # It tries to split on paragraphs first, then sentences, then characters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Each chunk ~1000 characters (~250 tokens)
            chunk_overlap=200,  # 200 character overlap to preserve context
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Split on these, in order
        )

        # Split all documents
        chunks = text_splitter.split_documents(documents)

        print(f"\n✂️  Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def create_embeddings_and_store(self, chunks: List[LangChainDocument]):
        """
        Create embeddings for all chunks and store in ChromaDB

        This is the KEY step in RAG - converting text to searchable vectors!

        Args:
            chunks: List of document chunks
        """
        print(f"\n🔢 Creating embeddings for {len(chunks)} chunks...")

        # Extract texts from chunks
        texts = [chunk.page_content for chunk in chunks]

        # Create embeddings using sentence-transformers
        # This happens locally on your machine!
        # Each text becomes a 384-dimensional vector
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32  # Process 32 chunks at a time
        )

        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(chunks))]
        metadatas = [chunk.metadata for chunk in chunks]

        # Store in ChromaDB
        # ChromaDB will use these embeddings for similarity search
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Stored {len(chunks)} chunks in ChromaDB")

    def build_vector_store(self):
        """
        Complete pipeline: Load → Split → Embed → Store

        Call this once to process all your documents!
        """
        # Check if already built
        if self.collection.count() > 0:
            print(f"⚠️  Vector store already contains {self.collection.count()} documents")
            print("   Delete './chroma_db' folder to rebuild from scratch")
            return

        # Step 1: Load documents
        documents = self.load_documents()

        if not documents:
            print("❌ No documents found!")
            return

        # Step 2: Split into chunks
        chunks = self.split_documents(documents)

        # Step 3: Create embeddings and store
        self.create_embeddings_and_store(chunks)

        print("\n🎉 Vector store built successfully!")

    def retrieve_relevant_chunks(self, question: str, n_results: int = 3) -> Dict:
        """
        Find the most relevant document chunks for a question

        This is where the magic happens!

        Args:
            question: User's question
            n_results: Number of relevant chunks to return

        Returns:
            Dictionary with chunks and their metadata
        """
        # Convert question to embedding
        question_embedding = self.embedding_model.encode([question])[0]

        # Query ChromaDB for similar chunks
        # ChromaDB uses cosine similarity to find closest vectors
        results = self.collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=n_results
        )

        # Format results nicely
        retrieved_chunks = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                chunk_info = {
                    'text': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                retrieved_chunks.append(chunk_info)

        return {
            'chunks': retrieved_chunks,
            'count': len(retrieved_chunks)
        }


# Example usage (for testing)
if __name__ == "__main__":
    # This runs when you execute: python src/rag_pipeline.py

    docs_dir = "/Users/raymondharrison/Desktop/primustech/data/unstructured"
    rag = RAGPipeline(docs_directory=docs_dir)

    # Build the vector store (only needed once!)
    print("\n" + "="*50)
    print("BUILDING VECTOR STORE")
    print("="*50)
    rag.build_vector_store()

    # Test retrieval
    print("\n" + "="*50)
    print("TESTING RETRIEVAL")
    print("="*50)

    test_question = "What are Microsoft's AI initiatives?"
    print(f"\nQuestion: {test_question}")

    results = rag.retrieve_relevant_chunks(test_question, n_results=2)

    print(f"\nFound {results['count']} relevant chunks:\n")
    for i, chunk in enumerate(results['chunks'], 1):
        print(f"Chunk {i}:")
        print(f"  Source: {chunk['metadata']['source']}")
        print(f"  Text preview: {chunk['text'][:200]}...")
        print(f"  Distance: {chunk['distance']:.4f}")
        print()
