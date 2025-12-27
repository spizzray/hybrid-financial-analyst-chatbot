"""
RAG Answer Generator - Generates answers using retrieved documents

This module:
1. Takes user questions
2. Retrieves relevant document chunks from vector store
3. Sends chunks + question to LLM
4. Generates answers based ONLY on retrieved context
5. Includes source citations for traceability

Key principle: Answers must be grounded in documents (no hallucination!)
"""

from typing import Dict, List
from rag_pipeline import RAGPipeline
from llm_handler import LLMHandler


class RAGAnswerGenerator:
    """Generates answers using Retrieval Augmented Generation"""

    def __init__(self, rag_pipeline: RAGPipeline, llm_handler: LLMHandler):
        """
        Initialize the RAG answer generator

        Args:
            rag_pipeline: The RAG pipeline for document retrieval
            llm_handler: The LLM for answer generation
        """
        self.rag = rag_pipeline
        self.llm = llm_handler

    def generate_answer(self, question: str, n_chunks: int = 3) -> Dict:
        """
        Generate an answer to the question using RAG

        Args:
            question: User's question
            n_chunks: Number of document chunks to retrieve

        Returns:
            Dictionary with:
                - success: Whether answer was generated
                - answer: The generated answer
                - sources: List of source documents used
                - chunks: The retrieved chunks (for debugging)
        """
        # Step 1: Retrieve relevant chunks
        print(f"🔍 Retrieving relevant chunks for: {question}")
        retrieval_result = self.rag.retrieve_relevant_chunks(question, n_results=n_chunks)

        if retrieval_result['count'] == 0:
            return {
                'success': False,
                'answer': "I couldn't find any relevant information in the documents to answer your question.",
                'sources': [],
                'chunks': []
            }

        chunks = retrieval_result['chunks']
        print(f"✅ Retrieved {len(chunks)} relevant chunks")

        # Step 2: Build context from chunks
        context = self._build_context(chunks)

        # Step 3: Generate answer using LLM + context
        answer = self._generate_answer_with_context(question, context)

        # Step 4: Extract sources for citation
        sources = self._extract_sources(chunks)

        return {
            'success': True,
            'answer': answer,
            'sources': sources,
            'chunks': chunks  # Include for debugging/transparency
        }

    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from retrieved chunks

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            source = chunk['metadata']['source']
            text = chunk['text']

            # Include page number if available (for PDFs)
            if 'page' in chunk['metadata']:
                page = chunk['metadata']['page']
                context_parts.append(f"[Source {i}: {source}, Page {page}]\n{text}")
            else:
                context_parts.append(f"[Source {i}: {source}]\n{text}")

        return "\n\n".join(context_parts)

    def _generate_answer_with_context(self, question: str, context: str) -> str:
        """
        Generate answer using LLM with retrieved context

        This is the core RAG step: LLM + Context = Grounded Answer

        Args:
            question: User's question
            context: Retrieved document context

        Returns:
            Generated answer
        """
        prompt = f"""You are a financial analyst assistant. Answer the user's question based ONLY on the provided context from company documents.

IMPORTANT RULES:
1. Use ONLY information from the context below
2. If the context doesn't contain the answer, say "I don't have enough information in the documents to answer this question"
3. Do NOT make up information or use external knowledge
4. Be specific and cite details from the documents when relevant
5. Keep answers concise (2-4 sentences)
6. If multiple companies are mentioned, compare them clearly

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {question}

ANSWER:"""

        response = self.llm.generate_response(prompt)
        return response.strip()

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """
        Extract unique sources from chunks

        Args:
            chunks: Retrieved chunks with metadata

        Returns:
            List of source information
        """
        sources = []
        seen = set()

        for chunk in chunks:
            source_name = chunk['metadata']['source']

            # Create unique identifier
            if 'page' in chunk['metadata']:
                page = chunk['metadata']['page']
                identifier = f"{source_name}_page_{page}"
                display = f"{source_name} (Page {page})"
            else:
                identifier = source_name
                display = source_name

            # Only add if not seen before
            if identifier not in seen:
                sources.append({
                    'file': source_name,
                    'page': chunk['metadata'].get('page'),
                    'display': display
                })
                seen.add(identifier)

        return sources


# Example usage (for testing)
if __name__ == "__main__":
    print("="*60)
    print("TESTING RAG ANSWER GENERATOR")
    print("="*60)

    # Initialize components
    try:
        docs_dir = "/Users/raymondharrison/Desktop/primustech/data/unstructured"
        rag_pipeline = RAGPipeline(docs_directory=docs_dir)
        llm = LLMHandler()
        rag_gen = RAGAnswerGenerator(rag_pipeline=rag_pipeline, llm_handler=llm)
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        exit(1)

    # Test questions
    test_questions = [
        "What are Microsoft's AI initiatives?",
        "What are NVIDIA's growth drivers?",
        "What challenges does Apple face?",
        "What is Meta's metaverse strategy?",
        "What are Google's cloud computing plans?",
    ]

    print("\n")
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print("="*60)

        result = rag_gen.generate_answer(question)

        if result['success']:
            print(f"✅ Success!")
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nSources:")
            for source in result['sources']:
                print(f"  - {source['display']}")
        else:
            print(f"❌ No relevant documents found")
            print(f"Response: {result['answer']}")

    print("\n✅ All tests completed!")
