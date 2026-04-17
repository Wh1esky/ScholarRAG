"""
LLM Client - 大语言模型调用封装

支持:
- OpenAI GPT-4/GPT-3.5
- Anthropic Claude-3
- OpenAI 兼容接口 (Azure, Local models, etc.)

使用方法:
    from src.rag.llm_client import LLMClient

    # OpenAI
    client = LLMClient(model_name="gpt-4", api_key="sk-...")

    # Claude
    client = LLMClient(
        model_name="claude-3-sonnet-20240229",
        api_key="sk-ant-...",
        provider="anthropic"
    )

    # 本地模型
    client = LLMClient(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        api_key="empty",
        base_url="http://localhost:8000/v1",
        provider="openai_compatible"
    )
"""

import os
import time
from typing import List, Optional, Dict, Any, Generator, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json

import requests


@dataclass
class LLMResponse:
    """LLM 响应结构"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    raw_response: Optional[Dict] = None
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.usage.get('total_tokens', 0)

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get('prompt_tokens', 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get('completion_tokens', 0)


class BaseLLMClient(ABC):
    """LLM Client 基类"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """生成单个回复"""
        pass

    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> List[LLMResponse]:
        """批量生成回复"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API 客户端"""

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key.")

    def _make_request(self, payload: Dict) -> Dict:
        """发送 API 请求"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"请求失败，{wait_time}秒后重试... ({attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"API 请求失败: {str(e)}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        # 正确使用 Chat API：system prompt 作为 system role
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        if stop:
            payload["stop"] = stop

        start_time = time.time()

        try:
            response = self._make_request(payload)
            latency_ms = (time.time() - start_time) * 1000

            content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})

            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                latency_ms=latency_ms,
                raw_response=response
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                error=str(e)
            )

    def chat_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """使用消息列表进行对话"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        if stop:
            payload["stop"] = stop

        start_time = time.time()

        try:
            response = self._make_request(payload)
            latency_ms = (time.time() - start_time) * 1000

            content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})

            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                latency_ms=latency_ms,
                raw_response=response
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                error=str(e)
            )

    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> List[LLMResponse]:
        """批量生成 (使用批量 API 或顺序调用)"""
        results = []
        for prompt in prompts:
            response = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            results.append(response)
        return results


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API 客户端"""

    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = "https://api.anthropic.com/v1"
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key.")

    def _make_request(self, payload: Dict) -> Dict:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }

        endpoint = f"{self.base_url}/messages"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"请求失败，{wait_time}秒后重试... ({attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"API 请求失败: {str(e)}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        if stop:
            payload["stop_sequences"] = stop

        start_time = time.time()

        try:
            response = self._make_request(payload)
            latency_ms = (time.time() - start_time) * 1000

            content = response["content"][0]["text"]
            usage = {
                "input_tokens": response["usage"]["input_tokens"],
                "output_tokens": response["usage"]["output_tokens"],
                "total_tokens": response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
            }

            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                latency_ms=latency_ms,
                raw_response=response
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                error=str(e)
            )

    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> List[LLMResponse]:
        results = []
        for prompt in prompts:
            response = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            results.append(response)
        return results


class LLMClient:
    """
    统一的 LLM 客户端

    支持多种 Provider:
    - openai: OpenAI GPT-4/GPT-3.5
    - anthropic: Anthropic Claude
    - openai_compatible: 兼容 OpenAI 接口的模型
    """

    PROVIDER_MODELS = {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
        "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    }

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.model_name = model_name
        self.provider = provider or self._detect_provider(model_name)

        if self.provider == "openai":
            self._client = OpenAIClient(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries
            )
        elif self.provider == "anthropic":
            self._client = AnthropicClient(
                model_name=model_name,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries
            )
        elif self.provider == "openai_compatible":
            self._client = OpenAIClient(
                model_name=model_name,
                api_key=api_key or "empty",
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _detect_provider(self, model_name: str) -> str:
        """自动检测 provider"""
        if model_name.startswith("claude"):
            return "anthropic"
        elif any(m in model_name for m in self.PROVIDER_MODELS["openai"]):
            return "openai"
        elif any(m in model_name for m in self.PROVIDER_MODELS["anthropic"]):
            return "anthropic"
        else:
            return "openai_compatible"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        return self._client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            **kwargs
        )

    def chat_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        if isinstance(self._client, OpenAIClient):
            return self._client.chat_generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs
            )
        elif isinstance(self._client, AnthropicClient):
            combined_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            return self._client.generate(
                prompt=combined_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs
            )
        else:
            raise NotImplementedError("chat_generate not supported for this provider")

    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> List[LLMResponse]:
        return self._client.batch_generate(
            prompts=prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider}, model={self.model_name})"


def create_llm_client(
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs
) -> LLMClient:
    """
    创建 LLM 客户端的工厂函数

    Args:
        model_name: 模型名称
        provider: Provider 类型 ('openai', 'anthropic', 'openai_compatible')
        **kwargs: 其他参数

    Returns:
        LLMClient 实例
    """
    model_name = model_name or os.environ.get("LLM_MODEL", "gpt-4")
    return LLMClient(model_name=model_name, provider=provider, **kwargs)


if __name__ == "__main__":
    print("=" * 50)
    print("LLM Client 测试")
    print("=" * 50)

    import os
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        print("警告: 未设置 OPENAI_API_KEY 环境变量")
        print("测试将使用模拟模式")

        class MockLLMResponse(LLMResponse):
            def __init__(self):
                super().__init__(
                    content="这是模拟回复。由于未设置 API Key，无法调用真实的 LLM。",
                    model="mock",
                    usage={"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
                    latency_ms=0.0
                )

        print("\n测试 Prompt:")
        print("-" * 40)
        print("请解释什么是 RAG (Retrieval-Augmented Generation)?")
        print("-" * 40)
        print("\n模拟回复:")
        print(MockLLMResponse().content)
    else:
        print(f"使用模型: {os.environ.get('LLM_MODEL', 'gpt-4')}")
        client = create_llm_client()

        response = client.generate(
            prompt="请用一句话解释什么是 RAG?",
            temperature=0.7,
            max_tokens=100
        )

        print(f"\n回复: {response.content}")
        print(f"使用 Token: {response.total_tokens}")
        print(f"延迟: {response.latency_ms:.2f}ms")
