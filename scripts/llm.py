"""API access to Claude Opus 4.6 and GPT-5.4 through Azure, with a response cache.

Environment variables:
  AZURE_AI_API_KEY, CLAUDE_BASE_URL        Claude Opus 4.6 (Azure AI Foundry)
  AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT   GPT-5.4 (Azure OpenAI)
"""
import json
import os
import time


class LLM:
    def __init__(self, name):
        self.name = name
        if name == "claude-opus-4.6":
            from anthropic import AnthropicFoundry
            self.client = AnthropicFoundry(api_key=os.environ["AZURE_AI_API_KEY"],
                                           base_url=os.environ["CLAUDE_BASE_URL"])
            self.model = "claude-opus-4-6"
        elif name == "gpt-5.4":
            from openai import AzureOpenAI
            self.client = AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                      azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                                      api_version="2024-12-01-preview")
            self.model = "gpt-5.4"
        else:
            raise ValueError(f"unknown model: {name}")

    def __call__(self, prompt, max_tokens, temperature=None):
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.name.startswith("claude"):
            r = self.client.messages.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                **kwargs)
            text = "".join(b.text for b in r.content if hasattr(b, "text"))
        else:
            r = self.client.chat.completions.create(
                model=self.model, max_completion_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}], **kwargs)
            text = r.choices[0].message.content or ""
        return text.strip()


class CachedLLM:
    """Calls a model once per key and stores the replies in a JSONL file."""

    def __init__(self, name, cache_path, retries=4):
        self.llm = None
        self.name = name
        self.retries = retries
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.cache[r["k"]] = r["v"]
                    except json.JSONDecodeError:
                        pass
        self.file = open(cache_path, "a")

    def ask(self, key, prompt, max_tokens, temperature=None):
        """Return the reply, or None if every attempt failed."""
        if key in self.cache:
            return self.cache[key]
        if self.llm is None:
            self.llm = LLM(self.name)
        reply = None
        for attempt in range(self.retries):
            try:
                reply = self.llm(prompt, max_tokens, temperature)
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        if reply is not None:
            self.file.write(json.dumps({"k": key, "v": reply}) + "\n")
            self.file.flush()
            self.cache[key] = reply
        return reply
