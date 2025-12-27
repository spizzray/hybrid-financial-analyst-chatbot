"""
Router - Decides whether to use SQL or RAG

This is the BRAIN of our hybrid system!

The router analyzes user questions and decides:
- SQL: For numerical data, metrics, comparisons (from database)
- RAG: For strategic insights, qualitative information (from documents)

This is one of the key evaluation criteria in the assignment!
"""

from typing import Literal, Dict
from llm_handler import LLMHandler


class QueryRouter:
    """Routes questions to SQL or RAG based on content"""

    def __init__(self, llm_handler: LLMHandler):
        """
        Initialize the router

        Args:
            llm_handler: The LLM handler for making routing decisions
        """
        self.llm = llm_handler

        # Define the routing prompt
        # This is CRITICAL - good prompt engineering makes or breaks the system!
        self.routing_prompt_template = """You are a query router for a financial analysis system.

You have access to TWO data sources:

1. **SQL DATABASE** - Contains structured numerical data:
   - Company names, tickers, sectors
   - Market cap, P/E ratios
   - Revenue and net income (2023)

   Use SQL for questions about:
   - Specific numbers/metrics (revenue, market cap, P/E ratio, etc.)
   - Comparisons between companies (which has higher revenue?)
   - Filtering/sorting (all tech companies, highest revenue, etc.)
   - Calculations (sum, average, difference)

2. **DOCUMENT DATABASE (RAG)** - Contains unstructured text:
   - Earnings call transcripts
   - 10-K reports
   - Strategic initiatives, AI plans
   - Management commentary
   - Future outlook, risks, opportunities

   Use RAG for questions about:
   - Strategy, plans, initiatives
   - Qualitative information (why, how, what challenges)
   - Future outlook
   - Management commentary
   - Technology details, product launches
   - Risks and opportunities

IMPORTANT RULES:
- If question asks about NUMBERS/METRICS → Choose SQL
- If question asks about STRATEGY/INSIGHTS → Choose RAG
- If question needs BOTH → Choose SQL (we'll enhance with RAG later if needed)
- If unclear → Prefer SQL for factual data, RAG for explanations

USER QUESTION: {question}

Respond with ONLY ONE WORD: either "SQL" or "RAG"
"""

    def route_question(self, question: str) -> Dict[str, str]:
        """
        Determine whether a question should use SQL or RAG

        Args:
            question: The user's question

        Returns:
            Dictionary with:
                - route: "sql" or "rag"
                - reasoning: Why this route was chosen (for debugging)
        """
        # Create the prompt
        prompt = self.routing_prompt_template.format(question=question)

        # Ask the LLM to route
        response = self.llm.generate_response(prompt).strip().upper()

        # Parse the response
        if "SQL" in response:
            route = "sql"
            reasoning = "Question requires structured numerical data from database"
        elif "RAG" in response:
            route = "rag"
            reasoning = "Question requires qualitative insights from documents"
        else:
            # Fallback: Use simple heuristics
            route = self._fallback_routing(question)
            reasoning = f"LLM routing failed, used fallback heuristic: {route}"

        return {
            "route": route,
            "reasoning": reasoning,
            "llm_response": response
        }

    def _fallback_routing(self, question: str) -> Literal["sql", "rag"]:
        """
        Fallback heuristic routing if LLM fails

        This is a simple keyword-based approach
        Not as good as LLM, but reliable backup

        Args:
            question: The user's question

        Returns:
            "sql" or "rag"
        """
        question_lower = question.lower()

        # Keywords that suggest SQL
        sql_keywords = [
            "revenue", "income", "profit", "earnings",
            "market cap", "pe ratio", "p/e ratio",
            "how much", "how many",
            "compare", "comparison",
            "highest", "lowest", "largest", "smallest",
            "total", "sum", "average",
            "list", "show me", "what is the",
            "sector", "ticker"
        ]

        # Keywords that suggest RAG
        rag_keywords = [
            "strategy", "strategic", "initiative", "plan",
            "ai", "artificial intelligence", "machine learning",
            "outlook", "future", "forecast",
            "challenge", "risk", "opportunity",
            "why", "how does", "explain",
            "announcement", "announced",
            "growth driver", "headwind", "tailwind",
            "management", "commentary", "mentioned",
            "technology", "product", "service"
        ]

        # Count keyword matches
        sql_score = sum(1 for keyword in sql_keywords if keyword in question_lower)
        rag_score = sum(1 for keyword in rag_keywords if keyword in question_lower)

        # Make decision
        if sql_score > rag_score:
            return "sql"
        elif rag_score > sql_score:
            return "rag"
        else:
            # Tie or no matches: default to SQL (safer for factual questions)
            return "sql"


# Example usage (for testing)
if __name__ == "__main__":
    print("="*60)
    print("TESTING QUERY ROUTER")
    print("="*60)

    # Initialize LLM and router
    try:
        llm = LLMHandler()
        router = QueryRouter(llm)
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        print("\n📝 Make sure your .env file has GOOGLE_API_KEY set!")
        exit(1)

    # Test questions
    test_questions = [
        # Should route to SQL:
        "What is Tesla's revenue?",
        "Compare Apple and Microsoft's market cap",
        "Which company has the highest P/E ratio?",
        "Show me all technology companies",

        # Should route to RAG:
        "What are Microsoft's AI initiatives?",
        "What are the growth drivers for NVIDIA?",
        "What challenges does Apple face?",
        "What did Meta announce about the metaverse?",

        # Edge cases:
        "Tell me about Apple",  # Ambiguous
        "Apple revenue and strategy",  # Mixed
    ]

    print("\n")
    for i, question in enumerate(test_questions, 1):
        print(f"{i}. Question: \"{question}\"")
        result = router.route_question(question)
        print(f"   Route: {result['route'].upper()}")
        print(f"   Reasoning: {result['reasoning']}")
        print()

    print("✅ Routing tests completed!")
