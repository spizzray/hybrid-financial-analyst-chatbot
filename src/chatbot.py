"""
Hybrid Financial Analyst Chatbot - Main Orchestrator

This module brings together all components:
- Router (decides SQL vs RAG)
- SQL Query Generator (for database queries)
- RAG Answer Generator (for document search)
- Conversation memory

This is the main interface that external applications (like Streamlit) will use.
"""

from typing import Dict, Optional
from database import FinancialDatabase
from rag_pipeline import RAGPipeline
from llm_handler import LLMHandler
from router import QueryRouter
from sql_query_generator import SQLQueryGenerator
from rag_answer_generator import RAGAnswerGenerator


class FinancialChatbot:
    """
    Main chatbot orchestrator that coordinates all components

    This is the "brain" that:
    1. Routes questions to the right data source
    2. Executes queries (SQL or RAG)
    3. Formats responses with traceability
    4. Manages conversation history
    """

    def __init__(
        self,
        database: FinancialDatabase,
        rag_pipeline: RAGPipeline,
        llm_handler: LLMHandler
    ):
        """
        Initialize the chatbot with all required components

        Args:
            database: DuckDB database instance
            rag_pipeline: RAG pipeline for document retrieval
            llm_handler: LLM handler for API calls
        """
        self.db = database
        self.rag = rag_pipeline
        self.llm = llm_handler

        # Initialize all subcomponents
        self.router = QueryRouter(llm_handler=llm_handler)
        self.sql_generator = SQLQueryGenerator(database=database, llm_handler=llm_handler)
        self.rag_generator = RAGAnswerGenerator(rag_pipeline=rag_pipeline, llm_handler=llm_handler)

        # Start chat session for conversation memory
        self.llm.start_chat()

        print("✅ Financial Chatbot initialized successfully!")

    def ask(self, question: str) -> Dict:
        """
        Main method to ask the chatbot a question

        This orchestrates the entire flow:
        1. Route the question (SQL or RAG)
        2. Execute the appropriate query
        3. Format the response
        4. Manage conversation history

        Args:
            question: User's question

        Returns:
            Dictionary with:
                - answer: The response text
                - route: Which path was taken ("sql" or "rag")
                - sources: Source information (SQL query or document citations)
                - success: Whether the query succeeded
                - metadata: Additional info (attempts, reasoning, etc.)
        """
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("="*60)

        # Step 1: Route the question
        routing_result = self.router.route_question(question)
        route = routing_result['route']
        reasoning = routing_result['reasoning']

        print(f"🧭 Route: {route.upper()}")
        print(f"📝 Reasoning: {reasoning}")

        # Step 2: Execute based on route
        if route == "sql":
            result = self._execute_sql_path(question)
        else:  # route == "rag"
            result = self._execute_rag_path(question)

        # Step 3: Manage conversation history
        self._update_conversation_history(question, result['answer'])

        # Step 4: Return formatted result
        return result

    def _execute_sql_path(self, question: str) -> Dict:
        """
        Execute SQL path: Generate and run SQL query

        Args:
            question: User's question

        Returns:
            Formatted result dictionary
        """
        print("\n🔄 Executing SQL path...")

        # Generate and execute SQL
        sql_result = self.sql_generator.generate_and_execute(question)

        if sql_result['success']:
            print("✅ SQL query succeeded!")

            return {
                'answer': sql_result['natural_language_answer'],
                'route': 'sql',
                'sources': {
                    'type': 'database',
                    'sql_query': sql_result['sql_query'],
                    'row_count': len(sql_result['data']) if sql_result['data'] else 0
                },
                'success': True,
                'metadata': {
                    'attempts': sql_result['attempts'],
                    'raw_data': sql_result['data']
                }
            }
        else:
            print("❌ SQL query failed")

            return {
                'answer': sql_result['natural_language_answer'],
                'route': 'sql',
                'sources': {
                    'type': 'database',
                    'sql_query': sql_result['sql_query'],
                    'error': sql_result['error']
                },
                'success': False,
                'metadata': {
                    'attempts': sql_result['attempts'],
                    'error': sql_result['error']
                }
            }

    def _execute_rag_path(self, question: str) -> Dict:
        """
        Execute RAG path: Retrieve documents and generate answer

        Args:
            question: User's question

        Returns:
            Formatted result dictionary
        """
        print("\n📚 Executing RAG path...")

        # Retrieve and generate answer
        rag_result = self.rag_generator.generate_answer(question)

        if rag_result['success']:
            print("✅ RAG query succeeded!")

            return {
                'answer': rag_result['answer'],
                'route': 'rag',
                'sources': {
                    'type': 'documents',
                    'files': rag_result['sources']
                },
                'success': True,
                'metadata': {
                    'chunk_count': len(rag_result['chunks']),
                    'chunks': rag_result['chunks']  # For debugging
                }
            }
        else:
            print("❌ RAG query failed (no relevant documents)")

            return {
                'answer': rag_result['answer'],
                'route': 'rag',
                'sources': {
                    'type': 'documents',
                    'files': []
                },
                'success': False,
                'metadata': {}
            }

    def _update_conversation_history(self, question: str, answer: str):
        """
        Update conversation history and manage sliding window

        Args:
            question: User's question
            answer: Chatbot's answer
        """
        # Conversation history is managed internally by LLM handler's chat session
        # This enables follow-up questions with context

        # Manage history to prevent context window overflow
        self.llm.manage_history(max_messages=20, keep_recent=10)

    def get_conversation_history(self) -> list:
        """
        Get the conversation history

        Returns:
            List of messages in the conversation
        """
        return self.llm.get_chat_history()

    def clear_conversation(self):
        """Clear the conversation history and start fresh"""
        self.llm.clear_chat()
        self.llm.start_chat()
        print("🗑️  Conversation history cleared")


# Example usage (for testing)
if __name__ == "__main__":
    import os

    print("="*60)
    print("TESTING HYBRID FINANCIAL CHATBOT")
    print("="*60)

    # Initialize all components
    try:
        # Database
        csv_path = "/Users/raymondharrison/Desktop/primustech/data/financial_data.csv"
        db = FinancialDatabase(csv_path)

        # RAG Pipeline
        docs_dir = "/Users/raymondharrison/Desktop/primustech/data/unstructured"
        rag = RAGPipeline(docs_directory=docs_dir)

        # LLM Handler
        llm = LLMHandler()

        # Chatbot
        chatbot = FinancialChatbot(database=db, rag_pipeline=rag, llm_handler=llm)

    except Exception as e:
        print(f"❌ Error initializing: {e}")
        exit(1)

    # Test questions (mix of SQL and RAG)
    test_questions = [
        # SQL questions
        "What is Apple's revenue?",
        "Which company has the highest market cap?",
        "Compare revenue of Apple and Microsoft",

        # RAG questions
        "What are Microsoft's AI initiatives?",
        "What are NVIDIA's growth drivers?",
        "What challenges does Apple face?",
    ]

    print("\n" + "="*60)
    print("RUNNING TEST QUESTIONS")
    print("="*60)

    for i, question in enumerate(test_questions, 1):
        result = chatbot.ask(question)

        print(f"\n📊 Result {i}:")
        print(f"Answer: {result['answer']}")
        print(f"Route: {result['route'].upper()}")
        print(f"Success: {result['success']}")

        if result['route'] == 'sql' and result['success']:
            print(f"SQL Query: {result['sources']['sql_query']}")
        elif result['route'] == 'rag' and result['success']:
            print(f"Sources: {[s['display'] for s in result['sources']['files']]}")

        print()

    print("="*60)
    print("✅ All tests completed!")
    print("="*60)

    # Clean up
    db.close()
