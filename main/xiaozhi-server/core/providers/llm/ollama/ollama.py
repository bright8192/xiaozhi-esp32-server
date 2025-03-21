from config.logger import setup_logging
import json
import requests
from core.providers.llm.base import LLMProviderBase

TAG = __name__
logger = setup_logging()


class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        self.model_name = config.get("model_name")
        self.base_url = config.get("base_url", "http://localhost:11434")
        # Remove trailing slash if present
        self.base_url = self.base_url.rstrip('/')
        logger.bind(tag=TAG).info(f"Initializing Ollama with base_url: {self.base_url}")

    def _convert_messages_to_prompt(self, messages):
        """Convert OpenAI message format to Ollama prompt format"""
        prompt = ""
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'system':
                prompt += f"System: {content}\n"
            elif role == 'user':
                prompt += f"Human: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        return prompt.strip()

    def response(self, session_id, dialogue):
        try:
            prompt = self._convert_messages_to_prompt(dialogue)
            logger.bind(tag=TAG).debug(f"Sending request to Ollama. Model: {self.model_name}, Prompt: {prompt}")

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True
            )

            response.raise_for_status()
            is_active = True

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        content = chunk.get('response', '')
                        if content:
                            if '<think>' in content:
                                is_active = False
                                content = content.split('<think>')[0]
                            if '</think>' in content:
                                is_active = True
                                content = content.split('</think>')[-1]
                            if is_active:
                                yield content
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"Error processing chunk: {e}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in Ollama response generation: {e}")
            yield "【Ollama服务响应异常】"

    def response_with_functions(self, session_id, dialogue, functions=None):
        try:
            prompt = self._convert_messages_to_prompt(dialogue)
            logger.bind(tag=TAG).debug(
                f"Sending function call request to Ollama. Model: {self.model_name}, Prompt: {prompt}")

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "functions": functions
                    }
                },
                stream=True
            )

            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        content = chunk.get('response', '')
                        # Since Ollama doesn't natively support function calls,
                        # we'll yield the content without tool calls
                        yield content, None
                    except Exception as e:
                        logger.bind(tag=TAG).error(f"Error processing chunk: {e}")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in Ollama function call: {e}")
            yield {"type": "content", "content": f"【Ollama服务响应异常: {str(e)}】"}
