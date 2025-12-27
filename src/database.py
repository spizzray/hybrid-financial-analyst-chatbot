"""
Database Module - Handles DuckDB operations

This module:
1. Loads CSV data into DuckDB (an in-memory SQL database)
2. Executes SQL queries safely
3. Returns results in a readable format

Think of DuckDB as Excel with superpowers - you can query data using SQL!
"""

import duckdb
import pandas as pd
from pathlib import Path


class FinancialDatabase:
    """Manages the DuckDB database for financial data"""

    def __init__(self, csv_path: str):
        """
        Initialize the database and load CSV data

        Args:
            csv_path: Path to the financial_data.csv file
        """
        # Create an in-memory DuckDB connection
        # ':memory:' means data lives in RAM (fast but not persistent)
        self.conn = duckdb.connect(':memory:')
        self.csv_path = csv_path
        self._load_data()

    def _load_data(self):
        """
        Load CSV data into DuckDB

        DuckDB can directly read CSV files and create tables!
        This is one of its superpowers.
        """
        try:
            # Create a table named 'financial_data' from the CSV
            # DuckDB automatically infers column types (integers, floats, strings)
            query = f"""
            CREATE TABLE financial_data AS
            SELECT * FROM read_csv_auto('{self.csv_path}')
            """
            self.conn.execute(query)
            print(f"✅ Loaded data from {self.csv_path}")

            # Let's verify it worked by counting rows
            row_count = self.conn.execute("SELECT COUNT(*) FROM financial_data").fetchone()[0]
            print(f"📊 Database contains {row_count} companies")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def execute_query(self, sql_query: str) -> dict:
        """
        Execute a SQL query and return results

        Args:
            sql_query: The SQL query to execute (e.g., "SELECT * FROM financial_data")

        Returns:
            Dictionary with:
                - success: Boolean indicating if query worked
                - data: Results as a list of dictionaries (or None if error)
                - error: Error message (or None if success)
                - row_count: Number of rows returned
        """
        try:
            # Execute the query
            result = self.conn.execute(sql_query)

            # Fetch all results
            rows = result.fetchall()

            # Get column names
            columns = [desc[0] for desc in result.description]

            # Convert to list of dictionaries for easier handling
            # Example: [{'company_name': 'Apple', 'revenue': 383.29}, ...]
            data = [dict(zip(columns, row)) for row in rows]

            return {
                'success': True,
                'data': data,
                'error': None,
                'row_count': len(data)
            }

        except Exception as e:
            # If SQL query fails, return error information
            # This is crucial for our LLM to learn from mistakes!
            return {
                'success': False,
                'data': None,
                'error': str(e),
                'row_count': 0
            }

    def get_schema(self) -> str:
        """
        Get the database schema (table structure)

        This is CRITICAL for Text-to-SQL!
        The LLM needs to know what columns exist to write correct queries.

        Returns:
            A formatted string describing the table structure
        """
        schema_query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'financial_data'
        ORDER BY ordinal_position
        """

        result = self.conn.execute(schema_query).fetchall()

        # Format schema in a human-readable way
        schema_str = "Table: financial_data\nColumns:\n"
        for col_name, col_type in result:
            schema_str += f"  - {col_name} ({col_type})\n"

        return schema_str

    def get_sample_data(self, limit: int = 3) -> pd.DataFrame:
        """
        Get sample rows from the database

        Useful for showing the LLM examples of actual data

        Args:
            limit: Number of sample rows to return

        Returns:
            Pandas DataFrame with sample data
        """
        query = f"SELECT * FROM financial_data LIMIT {limit}"
        return self.conn.execute(query).fetchdf()

    def close(self):
        """Close the database connection"""
        self.conn.close()


# Example usage (for testing)
if __name__ == "__main__":
    # This code runs when you execute: python src/database.py

    csv_path = "/Users/raymondharrison/Desktop/primustech/data/financial_data.csv"
    db = FinancialDatabase(csv_path)

    print("\n📋 Database Schema:")
    print(db.get_schema())

    print("\n📊 Sample Data:")
    print(db.get_sample_data())

    print("\n🔍 Test Query: Get all tech companies")
    result = db.execute_query("SELECT company_name, revenue_2023_billions FROM financial_data WHERE sector = 'Technology'")

    if result['success']:
        print(f"Found {result['row_count']} companies:")
        for row in result['data']:
            print(f"  - {row['company_name']}: ${row['revenue_2023_billions']}B revenue")
    else:
        print(f"Error: {result['error']}")

    db.close()
