# Test Questions for Assignment Demo

This document provides a comprehensive list of test questions to demonstrate all features of the Hybrid Financial Analyst Chatbot.

---

## 🗄️ Part 1: SQL Database Queries (Structured Data)

### Test 1.1: Simple Data Retrieval
**Purpose**: Show basic SQL query generation

```
Q: What is Apple's revenue?
Expected: Returns $383.29 billion
Shows: SQL query in sources
```

```
Q: What is Microsoft's market cap?
Expected: Returns market cap value
Shows: SQL query in sources
```

### Test 1.2: Comparisons
**Purpose**: Demonstrate JOIN/multiple row queries

```
Q: Compare the revenue of Apple and Microsoft
Expected: Shows both companies' revenues side by side
Shows: SQL with WHERE clause or multiple conditions
```

```
Q: Which company has the highest P/E ratio?
Expected: Identifies company with max P/E ratio
Shows: SQL with ORDER BY or MAX function
```

### Test 1.3: Aggregations
**Purpose**: Show SQL aggregation functions

```
Q: What is the average revenue of all companies?
Expected: Calculates and returns average
Shows: SQL with AVG() function
```

```
Q: How many technology sector companies are in the database?
Expected: Returns count
Shows: SQL with COUNT() and WHERE
```

### Test 1.4: Filtering
**Purpose**: Demonstrate WHERE clauses

```
Q: Show me all companies in the technology sector
Expected: Lists all tech companies
Shows: SQL with WHERE sector = 'Technology'
```

```
Q: Which companies have revenue over $200 billion?
Expected: Filters companies by revenue threshold
Shows: SQL with WHERE revenue > 200
```

---

## 📚 Part 2: RAG Queries (Document/Strategy Questions)

### Test 2.1: Strategic Initiatives
**Purpose**: Show document retrieval for qualitative questions

```
Q: What are Microsoft's AI initiatives?
Expected: Describes Azure AI, Copilot, OpenAI partnership, etc.
Shows: Document sources (microsoft.txt or microsoft.pdf)
```

```
Q: What is NVIDIA's AI strategy?
Expected: Describes GPU focus, data center growth, AI platforms
Shows: Document sources (nvidia.pdf)
```

### Test 2.2: Challenges and Risks
**Purpose**: Demonstrate retrieval of negative/risk information

```
Q: What challenges does Apple face?
Expected: Discusses supply chain, competition, regulatory issues
Shows: Document sources (apple.txt)
```

```
Q: What are the headwinds facing Meta?
Expected: Describes metaverse concerns, ad revenue challenges, etc.
Shows: Document sources (meta.pdf)
```

### Test 2.3: Company Outlook
**Purpose**: Show retrieval of forward-looking statements

```
Q: What is Google's growth strategy?
Expected: Discusses cloud expansion, AI investment, search innovation
Shows: Document sources (google.pdf)
```

```
Q: What are NVIDIA's future plans?
Expected: Discusses product roadmap, market expansion
Shows: Document sources (nvidia.pdf)
```

---

## 🔄 Part 3: Statefulness (Conversation Memory)

### Test 3.1: Follow-up Questions - SQL
**Purpose**: Demonstrate conversation memory with SQL queries

```
Step 1: What is Apple's revenue?
Expected: Returns $383.29 billion

Step 2: What about Microsoft?
Expected: Understands "Microsoft" refers to revenue question
Returns: Microsoft's revenue
Shows: Context preservation
```

```
Step 1: Which company has the highest market cap?
Expected: Returns company name

Step 2: What is their revenue?
Expected: Understands "their" refers to previous company
Shows: Pronoun resolution via context
```

### Test 3.2: Follow-up Questions - RAG
**Purpose**: Demonstrate conversation memory with document queries

```
Step 1: What are Microsoft's AI initiatives?
Expected: Describes Azure AI, Copilot, etc.

Step 2: What drove that growth?
Expected: Understands "that" refers to Microsoft's AI growth
Retrieves: Growth drivers from documents
```

```
Step 1: What challenges does Apple face?
Expected: Lists challenges

Step 2: How are they addressing these issues?
Expected: Understands context, retrieves Apple's response strategies
```

### Test 3.3: Mixed Follow-ups
**Purpose**: Show context switching between SQL and RAG

```
Step 1: What is NVIDIA's revenue?
Expected: Returns revenue (SQL)

Step 2: What's driving that growth?
Expected: Routes to RAG, retrieves growth drivers from documents
Shows: Intelligent routing with context
```

---

## 🧭 Part 4: Routing Logic (SQL vs RAG)

### Test 4.1: Ambiguous Questions (Tests Routing Intelligence)
**Purpose**: Show the system correctly identifies data type needed

```
Q: Tell me about Apple
Expected: Routes to RAG (qualitative overview)
Shows: Routing decision indicator
```

```
Q: How is Apple performing financially?
Expected: Could route to SQL (numbers) or RAG (analysis)
Shows: System's routing choice with reasoning
```

### Test 4.2: Keyword-based Routing
**Purpose**: Demonstrate fallback routing mechanism

```
Q: What is the income of Tesla?
Expected: Routes to SQL (financial metric keyword)
Shows: Fallback heuristic if LLM routing fails
```

```
Q: What is Apple's strategy for AI?
Expected: Routes to RAG (strategy keyword)
Shows: Keyword-based routing
```

---

## ⚠️ Part 5: Error Handling

### Test 5.1: SQL Error Recovery
**Purpose**: Show iterative refinement with retry logic

```
Q: What's Apple's revenues in 2023?
Note: "revenues" (plural) vs "revenue" in schema
Expected: First attempt may fail, system corrects and retries
Shows: Error detection and SQL correction
```

### Test 5.2: Missing Data
**Purpose**: Demonstrate graceful handling of unavailable data

```
Q: What is Tesla's revenue?
Expected: "I couldn't find data for Tesla. Available companies: [list]"
Shows: User-friendly error message
```

```
Q: What is Apple's profit margin?
Note: "profit margin" column doesn't exist
Expected: "Data not available. Available metrics: [list]"
Shows: Helpful suggestion of available data
```

### Test 5.3: Ambiguous RAG Queries
**Purpose**: Show handling when documents don't have answer

```
Q: What is Microsoft's employee count?
Expected: May not be in documents
Shows: "No relevant information found" with helpful message
```

---

## 📊 Part 6: Traceability (Source Attribution)

### Test 6.1: SQL Source Traceability
**Purpose**: Show SQL query transparency

```
Q: What is the average market cap?
Expected: Returns calculation result
**Click "View Sources"**
Shows: Exact SQL query that was executed
Shows: Number of rows returned
```

### Test 6.2: RAG Source Traceability
**Purpose**: Show document citation

```
Q: What are NVIDIA's AI products?
Expected: Lists AI products
**Click "View Sources"**
Shows: Source document names (nvidia.pdf)
Shows: Page numbers or document sections
```

---

## 🎯 Part 7: Edge Cases and Complex Queries

### Test 7.1: Multi-condition SQL
**Purpose**: Show complex query handling

```
Q: Which technology companies have revenue over $300 billion and P/E ratio under 30?
Expected: Filters by sector, revenue, and P/E ratio
Shows: Complex SQL with multiple WHERE conditions
```

### Test 7.2: Compound Questions
**Purpose**: Test handling of questions with multiple parts

```
Q: What is Apple's revenue and what are their AI initiatives?
Expected: May route to one source (SQL or RAG)
Shows: System's routing decision for compound questions
```

### Test 7.3: Negative Questions
**Purpose**: Show handling of questions with negations

```
Q: Which companies do NOT have a P/E ratio over 30?
Expected: Returns filtered list
Shows: SQL with NOT or inverse condition
```

---

## 📸 Screenshot Checklist

For comprehensive demo, capture screenshots showing:

### ✅ Must-Have Screenshots:

1. **SQL Query Success**
   - Question + Answer + SQL source visible
   - Example: "What is Apple's revenue?"

2. **RAG Query Success**
   - Question + Answer + Document sources visible
   - Example: "What are Microsoft's AI initiatives?"

3. **Conversation Memory**
   - 2-3 message exchange showing follow-up context
   - Example: "What is Microsoft's revenue?" → "What about Apple?"

4. **Routing Indicator**
   - Show routing decision (SQL or RAG)
   - Visible in UI or console output

5. **Error Handling**
   - User-friendly error message for missing data
   - Example: "What is Tesla's revenue?"

6. **Traceability - SQL**
   - Expanded "View Sources" showing SQL query
   - Example: Any SQL question with sources expanded

7. **Traceability - RAG**
   - Expanded "View Sources" showing document citations
   - Example: Any RAG question with sources expanded

8. **Comparison Query**
   - Multi-row result
   - Example: "Compare revenue of Apple and Microsoft"

9. **Aggregation Query**
   - Calculated result
   - Example: "What is the average revenue?"

10. **Complex RAG Query**
    - Detailed strategic question
    - Example: "What challenges does Meta face?"

---

## 🎬 Suggested Demo Flow (10 Screenshots)

If limited to ~10 screenshots, use this optimal sequence:

1. **SQL Simple**: "What is Apple's revenue?" (with SQL source visible)
2. **SQL Comparison**: "Compare revenue of Apple and Microsoft"
3. **SQL Aggregation**: "What is the average market cap?"
4. **RAG Strategy**: "What are Microsoft's AI initiatives?" (with doc sources visible)
5. **RAG Challenges**: "What challenges does Apple face?"
6. **Follow-up SQL**: Show 2-message exchange demonstrating context memory
7. **Follow-up RAG**: Show 2-message exchange demonstrating context memory
8. **Routing Decision**: Show routing indicator (SQL vs RAG choice)
9. **Error Handling**: "What is Tesla's revenue?" showing graceful failure
10. **Complex Query**: "Which companies have revenue over $200 billion?"

---

## 💡 Tips for Demo

1. **Clear Chat Between Major Tests**: Click "Clear Conversation" to reset for clean screenshots
2. **Expand Sources**: Always expand "View Sources" for traceability screenshots
3. **Check Console**: Terminal/console output shows routing decisions and debugging info
4. **Highlight Features**: Annotate screenshots to point out key features
5. **Show Variety**: Use different companies to demonstrate breadth
6. **Demonstrate Speed**: Note response times if impressive

---

## 📝 README Documentation Points

For each screenshot, document in README:
- **Question asked**
- **Feature demonstrated** (SQL, RAG, memory, routing, error handling, traceability)
- **What to observe** (source type, routing decision, context preservation, etc.)
- **Expected behavior**

---

## ⏱️ Estimated Testing Time

- Basic tests (1-7): ~5 minutes
- Conversation memory (3.1-3.3): ~3 minutes
- Edge cases (7.1-7.3): ~2 minutes
- **Total**: ~10-15 minutes for comprehensive testing

---

## 🔄 After Testing

Document in README:
- ✅ All features working as expected
- ⚠️ Any limitations encountered
- 📊 Success rates (e.g., "95% accurate routing")
- 💡 Interesting observations

Good luck with your demo! 🚀
