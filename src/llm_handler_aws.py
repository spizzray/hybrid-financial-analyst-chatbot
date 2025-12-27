"""
LLM Handler - AWS Bedrock Version

Supports AWS Bedrock with Claude models as an alternative to Google Gemini.
This allows using AWS credentials instead of Google API keys.
"""

import os
import json
import time
from typing import List, Dict, Optional
import boto3
from dotenv import load_dotenv


class LLMHandler:
    """Manages all interactions with AWS Bedrock LLM (Claude)"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        """
        Initialize the LLM handler with AWS Bedrock

        Args:
            api_key: Not used for Bedrock (uses AWS credentials)
            model_name: Which Bedrock model to use
                - anthropic.claude-3-haiku-20240307-v1:0: Fast, cheap (default)
                - anthropic.claude-3-sonnet-20240229-v1:0: Balanced
                - anthropic.claude-3-5-sonnet-20240620-v1:0: Most capable
        """
        # Load AWS credentials from .env file
        load_dotenv()

        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')

        if not aws_access_key or not aws_secret_key:
            raise ValueError(
                "AWS credentials not found! "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file"
            )

        # Initialize Bedrock client
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )

        self.model_id = model_name
        self.conversation_history = []

        print(f"✅ Initialized AWS Bedrock model: {model_name}")

    def generate_response(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Generate a response from the LLM

        Args:
            prompt: The user's question or instruction
            system_instruction: Optional system-level instructions

        Returns:
            The LLM's response as a string
        """
        try:
            # Prepare the request body for Claude
            messages = [{"role": "user", "content": prompt}]

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": 0.1,
                "messages": messages
            }

            if system_instruction:
                body["system"] = system_instruction

            # Call Bedrock
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            # Parse response
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']

        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {str(e)}"

    def start_chat(self, history: Optional[List[Dict]] = None):
        """
        Start a new chat session with conversation memory

        Args:
            history: Optional previous conversation history
        """
        if history:
            self.conversation_history = history
        else:
            self.conversation_history = []

        print("💬 Chat session started")

    def send_message(self, message: str) -> str:
        """
        Send a message in an ongoing chat session

        Args:
            message: User's message

        Returns:
            LLM's response
        """
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": message
            })

            # Prepare messages for Claude
            messages = []
            for msg in self.conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": 0.1,
                "messages": messages
            }

            # Call Bedrock
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )

            # Parse response
            response_body = json.loads(response['body'].read())
            assistant_message = response_body['content'][0]['text']

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

        except Exception as e:
            print(f"❌ Error sending message: {e}")
            return f"Error: {str(e)}"

    def get_chat_history(self) -> List[Dict]:
        """
        Get the current chat history

        Returns:
            List of messages in the conversation
        """
        return self.conversation_history

    def clear_chat(self):
        """Clear the current chat session"""
        self.conversation_history = []
        print("🗑️  Chat session cleared")

    def manage_history(self, max_messages: int = 20, keep_recent: int = 10):
        """
        Manage chat history using a sliding window approach

        Args:
            max_messages: Maximum number of messages before trimming
            keep_recent: How many recent messages to keep after trimming

        Returns:
            True if history was trimmed, False otherwise
        """
        if len(self.conversation_history) > max_messages:
            # Keep only the most recent messages
            self.conversation_history = self.conversation_history[-keep_recent:]
            print(f"📝 Trimmed history: Kept most recent {keep_recent} messages")
            return True
        return False

    def generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate response with automatic retry on failure

        Args:
            prompt: The prompt to send
            max_retries: Maximum number of retry attempts

        Returns:
            LLM response
        """
        for attempt in range(max_retries):
            try:
                return self.generate_response(prompt)
            except Exception as e:
                print(f"⚠️  Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    return f"Error after {max_retries} attempts: {str(e)}"
