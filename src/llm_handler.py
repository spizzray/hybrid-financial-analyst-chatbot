"""
LLM Handler - Manages interactions with Google Gemini API

This module:
1. Initializes connection to Google Gemini
2. Sends prompts and receives responses
3. Handles API errors and retries
4. Manages conversation history

Think of this as the "translator" that talks to the AI!
"""

import os
import time
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv


class LLMHandler:
    """Manages all interactions with Google Gemini LLM"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        """
        Initialize the LLM handler

        Args:
            api_key: Google API key (if None, loads from .env)
            model_name: Which Gemini model to use
                - gemini-2.0-flash: Fast, 1500 req/day free tier (default)
                - gemini-2.5-flash: Newer but only 20 req/day free tier
                - gemini-flash-latest: Always points to latest flash model
        """
        # Load API key from .env file if not provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('GOOGLE_API_KEY')

        if not api_key:
            raise ValueError(
                "Google API key not found! "
                "Set GOOGLE_API_KEY in .env file or pass as parameter"
            )

        # Configure the Google Generative AI library
        genai.configure(api_key=api_key)

        # Initialize the model
        # generation_config controls how the model generates responses
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.1,  # Low temperature = more deterministic/focused
                "top_p": 0.95,       # Nucleus sampling parameter
                "top_k": 40,         # Consider top 40 tokens
                "max_output_tokens": 2048,  # Maximum length of response
            }
        )

        # Store conversation history for multi-turn conversations
        # This enables follow-up questions!
        self.chat_session = None

        print(f"✅ Initialized Gemini model: {model_name}")

    def generate_response(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Generate a response from the LLM

        Args:
            prompt: The user's question or instruction
            system_instruction: Optional system-level instructions
                (e.g., "You are a financial analyst")

        Returns:
            The LLM's response as a string
        """
        try:
            # If system instruction provided, prepend it
            full_prompt = prompt
            if system_instruction:
                full_prompt = f"{system_instruction}\n\n{prompt}"

            # Send to Gemini and get response
            response = self.model.generate_content(full_prompt)

            # Extract text from response
            return response.text

        except Exception as e:
            # Handle API errors gracefully
            print(f"❌ Error generating response: {e}")
            return f"Error: {str(e)}"

    def start_chat(self, history: Optional[List[Dict]] = None):
        """
        Start a new chat session with conversation memory

        This allows follow-up questions like:
        User: "What is Apple's revenue?"
        Bot: "$383.29B"
        User: "What about Microsoft?"  <- Understands context!

        Args:
            history: Optional previous conversation history
        """
        # Convert history to Gemini's format if provided
        gemini_history = []
        if history:
            for msg in history:
                gemini_history.append({
                    "role": msg["role"],  # "user" or "model"
                    "parts": [msg["content"]]
                })

        # Start chat session
        self.chat_session = self.model.start_chat(history=gemini_history)
        print("💬 Chat session started")

    def send_message(self, message: str) -> str:
        """
        Send a message in an ongoing chat session

        Args:
            message: User's message

        Returns:
            LLM's response
        """
        if self.chat_session is None:
            # If no chat session, start one
            self.start_chat()

        try:
            response = self.chat_session.send_message(message)
            return response.text

        except Exception as e:
            print(f"❌ Error sending message: {e}")
            return f"Error: {str(e)}"

    def get_chat_history(self) -> List[Dict]:
        """
        Get the current chat history

        Returns:
            List of messages in the conversation
        """
        if self.chat_session is None:
            return []

        history = []
        for msg in self.chat_session.history:
            history.append({
                "role": msg.role,
                "content": msg.parts[0].text
            })

        return history

    def clear_chat(self):
        """Clear the current chat session"""
        self.chat_session = None
        print("🗑️  Chat session cleared")

    def manage_history(self, max_messages: int = 20, keep_recent: int = 10):
        """
        Manage chat history using a sliding window approach

        Instead of clearing ALL history, this keeps the most recent messages.
        This preserves context while preventing context window overflow!

        Example:
            History has 25 messages
            max_messages=20, keep_recent=10
            → Deletes messages 1-15, keeps messages 16-25

        Args:
            max_messages: Maximum number of messages before trimming
            keep_recent: How many recent messages to keep after trimming

        Returns:
            True if history was trimmed, False otherwise
        """
        if self.chat_session is None:
            return False

        current_history = self.get_chat_history()

        if len(current_history) > max_messages:
            # Keep only the most recent messages (sliding window!)
            recent_history = current_history[-keep_recent:]

            # Restart chat with recent history
            self.start_chat(history=recent_history)

            print(f"📝 Trimmed history: Kept most recent {keep_recent} messages")
            return True

        return False

    def generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate response with automatic retry on failure

        Useful for handling temporary API issues

        Args:
            prompt: The prompt to send
            max_retries: Maximum number of retry attempts

        Returns:
            LLM response
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text

            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # All retries exhausted
                    return f"Error after {max_retries} attempts: {str(e)}"


# Example usage (for testing)
if __name__ == "__main__":
    # This runs when you execute: python src/llm_handler.py

    print("="*50)
    print("TESTING LLM HANDLER")
    print("="*50)

    # Initialize handler
    try:
        llm = LLMHandler()
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\n📝 To test this module:")
        print("1. Get API key from: https://aistudio.google.com/app/apikey")
        print("2. Create .env file with: GOOGLE_API_KEY=your_key_here")
        exit(1)

    # Test 1: Simple query
    print("\n" + "-"*50)
    print("Test 1: Simple Query")
    print("-"*50)
    prompt = "What is 2 + 2? Answer in one sentence."
    response = llm.generate_response(prompt)
    print(f"Question: {prompt}")
    print(f"Response: {response}")

    # Test 2: Chat with context
    print("\n" + "-"*50)
    print("Test 2: Chat with Context")
    print("-"*50)
    llm.start_chat()

    msg1 = "My name is Alice. Remember this!"
    response1 = llm.send_message(msg1)
    print(f"User: {msg1}")
    print(f"Bot: {response1}")

    msg2 = "What is my name?"  # Should remember "Alice"
    response2 = llm.send_message(msg2)
    print(f"\nUser: {msg2}")
    print(f"Bot: {response2}")

    # Test 3: View history
    print("\n" + "-"*50)
    print("Test 3: Chat History")
    print("-"*50)
    history = llm.get_chat_history()
    print(f"Total messages: {len(history)}")
    for i, msg in enumerate(history):
        print(f"{i+1}. {msg['role']}: {msg['content'][:50]}...")

    # Test 4: Sliding window history management
    print("\n" + "-"*50)
    print("Test 4: Sliding Window History")
    print("-"*50)
    print("Simulating long conversation...")

    # Add many messages to exceed limit
    for i in range(3, 8):
        llm.send_message(f"Message number {i}")

    print(f"History now has: {len(llm.get_chat_history())} messages")

    # Manage history with sliding window
    llm.manage_history(max_messages=6, keep_recent=3)

    print(f"After sliding window: {len(llm.get_chat_history())} messages")
    print("Most recent messages:")
    for i, msg in enumerate(llm.get_chat_history()[-3:], 1):
        print(f"  {i}. {msg['role']}: {msg['content'][:40]}...")

    print("\n✅ All tests completed!")
