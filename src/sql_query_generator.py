"""
Text-to-SQL Generator - Converts natural language to SQL queries

This module:
1. Takes user questions in natural language
2. Generates SQL queries using an LLM
3. Executes queries against DuckDB
4. Handles errors and retries with corrections
5. Formats results for display

The key challenge: Getting the LLM to write CORRECT SQL!
"""

from typing import Dict
from database import FinancialDatabase
from llm_handler import LLMHandler


class SQLQueryGenerator:
    """Generates and executes SQL queries from natural language"""

    def __init__(self, database: FinancialDatabase, llm_handler: LLMHandler):
        """
        Initialize the SQL query generator

        Args:
            database: The DuckDB database instance
            llm_handler: The LLM for generating SQL
        """
        self.db = database
        self.llm = llm_handler

        # Get schema once (for all queries)
        self.schema = self.db.get_schema()

    def generate_and_execute(self, question: str, max_retries: int = 2) -> Dict:
        """
        Generate SQL from question and execute it

        This implements the "iterative refinement" pattern:
        1. Generate SQL
        2. Try to execute
        3. If error → Send error back to LLM → Retry
        4. Repeat until success or max retries

        Args:
            question: User's natural language question
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary with:
                - success: Whether query succeeded
                - data: Query results (if success)
                - sql_query: The SQL that was executed
                - natural_language_answer: Human-readable response
                - error: Error message (if failed)
        """
        sql_query = None
        last_error = None

        for attempt in range(max_retries + 1):
            # Generate SQL query
            if attempt == 0:
                # First attempt: Generate fresh SQL
                sql_query = self._generate_sql(question)
            else:
                # Retry: Generate SQL with error feedback
                print(f"⚠️  Attempt {attempt + 1}: Retrying with error feedback...")
                sql_query = self._generate_sql_with_error(question, sql_query, last_error)

            print(f"🔍 Generated SQL: {sql_query}")

            # Execute the query
            result = self.db.execute_query(sql_query)

            if result['success']:
                # Success! Format and return
                nl_answer = self._format_results(question, result['data'])

                return {
                    'success': True,
                    'data': result['data'],
                    'sql_query': sql_query,
                    'natural_language_answer': nl_answer,
                    'error': None,
                    'attempts': attempt + 1
                }
            else:
                # Query failed, store error for retry
                last_error = result['error']
                print(f"❌ SQL Error: {last_error}")

        # All attempts exhausted - format user-friendly error
        return {
            'success': False,
            'data': None,
            'sql_query': sql_query,
            'natural_language_answer': self._format_error_message(question, sql_query, last_error),
            'error': last_error,
            'attempts': max_retries + 1
        }

    def _generate_sql(self, question: str) -> str:
        """
        Generate SQL query from natural language question

        This is where prompt engineering is CRITICAL!

        Args:
            question: User's question

        Returns:
            SQL query as string
        """
        prompt = f"""You are an expert SQL query generator.

DATABASE SCHEMA:
{self.schema}

SAMPLE DATA:
{self.db.get_sample_data().to_string()}

INSTRUCTIONS:
1. Write a DuckDB SQL query to answer the user's question
2. Use ONLY columns that exist in the schema
3. Return ONLY the SQL query, nothing else
4. Do not include markdown formatting or explanations
5. Use proper SQL syntax for DuckDB

IMPORTANT NOTES:
- For company name matching, use exact names from the schema (e.g., "Apple Inc.", not "Apple")
- Use LIKE '%name%' for partial matches if user doesn't specify full name
- Always use proper column names (revenue_2023_billions, not just revenue)

USER QUESTION: {question}

SQL QUERY:"""

        response = self.llm.generate_response(prompt)

        # Clean up the response (remove markdown, extra whitespace)
        sql_query = response.strip()

        # Remove markdown code blocks if present
        if sql_query.startswith("```"):
            # Remove opening ```sql or ```
            sql_query = sql_query.split("\n", 1)[1] if "\n" in sql_query else sql_query[3:]
            # Remove closing ```
            if "```" in sql_query:
                sql_query = sql_query.rsplit("```", 1)[0]

        sql_query = sql_query.strip()

        # Remove "sql" keyword if present at the start
        if sql_query.lower().startswith("sql"):
            sql_query = sql_query[3:].strip()

        return sql_query

    def _generate_sql_with_error(self, question: str, failed_sql: str, error_message: str) -> str:
        """
        Generate corrected SQL based on previous error

        This is the "learning from mistakes" step!

        Args:
            question: Original user question
            failed_sql: The SQL that failed
            error_message: The error message from DuckDB

        Returns:
            Corrected SQL query
        """
        prompt = f"""You are an expert SQL query generator. Your previous query failed with an error.

DATABASE SCHEMA:
{self.schema}

USER QUESTION: {question}

YOUR PREVIOUS SQL (FAILED):
{failed_sql}

ERROR MESSAGE:
{error_message}

INSTRUCTIONS:
1. Analyze the error message
2. Fix the SQL query to address the error
3. Common issues:
   - Column name typos
   - Missing quotes around strings
   - Incorrect table name
   - Syntax errors
4. Return ONLY the corrected SQL query

CORRECTED SQL QUERY:"""

        response = self.llm.generate_response(prompt)

        # Clean up (same as _generate_sql)
        sql_query = response.strip()

        # Remove markdown code blocks if present
        if sql_query.startswith("```"):
            # Remove opening ```sql or ```
            sql_query = sql_query.split("\n", 1)[1] if "\n" in sql_query else sql_query[3:]
            # Remove closing ```
            if "```" in sql_query:
                sql_query = sql_query.rsplit("```", 1)[0]

        sql_query = sql_query.strip()

        # Remove "sql" keyword if present at the start
        if sql_query.lower().startswith("sql"):
            sql_query = sql_query[3:].strip()

        return sql_query

    def _format_results(self, question: str, data: list) -> str:
        """
        Convert SQL results into natural language answer

        Args:
            question: The original question
            data: Query results (list of dictionaries)

        Returns:
            Human-readable answer
        """
        if not data:
            return "No results found for your query."

        # Ask LLM to format results naturally
        prompt = f"""Convert these SQL query results into a natural language answer.

USER QUESTION: {question}

QUERY RESULTS:
{data}

INSTRUCTIONS:
1. Write a clear, concise answer to the user's question
2. Include specific numbers and company names
3. Format numbers nicely (e.g., $383.29 billion, not 383.29)
4. If multiple rows, present them in a readable way
5. Keep it brief (2-3 sentences max)

NATURAL LANGUAGE ANSWER:"""

        response = self.llm.generate_response(prompt)
        return response.strip()

    def _format_error_message(self, question: str, sql_query: str, error: str) -> str:
        """
        Format a user-friendly error message when SQL generation fails

        Instead of showing raw technical errors, this provides:
        1. Clear explanation of what went wrong
        2. Helpful suggestions for the user
        3. Optional SQL transparency for debugging

        Args:
            question: The original user question
            sql_query: The SQL that failed
            error: The error message from DuckDB

        Returns:
            User-friendly error message
        """
        error_lower = error.lower()

        # Case 1: Column or table doesn't exist
        if "does not exist" in error_lower or "not found" in error_lower:
            return (
                "I couldn't find that information in the database. "
                "Our database contains:\n\n"
                "📊 Available data:\n"
                "  • Company names, tickers, sectors\n"
                "  • Market capitalization\n"
                "  • P/E ratios\n"
                "  • Revenue (2023)\n"
                "  • Net income (2023)\n\n"
                "Try asking about one of these metrics, like:\n"
                "  • 'What is Apple's revenue?'\n"
                "  • 'Compare Microsoft and Alphabet's market cap'\n"
                "  • 'Which company has the highest P/E ratio?'"
            )

        # Case 2: Syntax error in SQL
        elif "syntax error" in error_lower or "parser" in error_lower:
            return (
                "I had trouble understanding your question. "
                "Could you rephrase it more simply?\n\n"
                "Examples of questions I understand well:\n"
                "  • 'What is [company]'s [metric]?'\n"
                "  • 'Compare [company1] and [company2]'\n"
                "  • 'Show me all [sector] companies'\n"
                "  • 'Which company has the highest [metric]?'"
            )

        # Case 3: Type mismatch or conversion error
        elif "type" in error_lower or "cast" in error_lower or "conversion" in error_lower:
            return (
                "I encountered a data type issue with your query. "
                "This usually happens when comparing incompatible values.\n\n"
                "Tips:\n"
                "  • For numeric comparisons, use numbers (e.g., 'revenue > 100')\n"
                "  • For text matching, use quotes (e.g., \"sector = 'Technology'\")\n"
                "  • Try rephrasing your question"
            )

        # Case 4: Division by zero or math error
        elif "division" in error_lower or "zero" in error_lower:
            return (
                "I encountered a mathematical error in my calculation. "
                "This might happen when dividing by zero or invalid operations.\n\n"
                "Try asking your question in a different way."
            )

        # Case 5: Generic error - show SQL for transparency
        else:
            return (
                f"I encountered an unexpected error while querying the database.\n\n"
                f"🔍 For transparency, here's what I tried:\n"
                f"SQL Query:\n{sql_query}\n\n"
                f"Error: {error}\n\n"
                f"This might be a limitation of my SQL generation. "
                f"Try rephrasing your question or asking something simpler."
            )


# Example usage (for testing)
if __name__ == "__main__":
    from pathlib import Path

    print("="*60)
    print("TESTING SQL QUERY GENERATOR")
    print("="*60)

    # Initialize components
    try:
        csv_path = "/Users/raymondharrison/Desktop/primustech/data/financial_data.csv"
        db = FinancialDatabase(csv_path)
        llm = LLMHandler()
        sql_gen = SQLQueryGenerator(database=db, llm_handler=llm)
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        exit(1)

    # Test questions
    test_questions = [
        "What is Apple's revenue?",
        "Which company has the highest market cap?",
        "Compare revenue of Apple and Microsoft",
        "Show me all technology companies",
        "What is the average P/E ratio of all companies?",
    ]

    print("\n")
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print("="*60)

        result = sql_gen.generate_and_execute(question)

        if result['success']:
            print(f"✅ Success! (Attempts: {result['attempts']})")
            print(f"\nSQL Query:\n{result['sql_query']}")
            print(f"\nAnswer:\n{result['natural_language_answer']}")
        else:
            print(f"❌ Failed after {result['attempts']} attempts")
            print(f"Error: {result['error']}")

    print("\n✅ All tests completed!")
    db.close()
