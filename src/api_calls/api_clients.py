#!/usr/bin/env python3

import os
import yaml
import jsonlines
import asyncio
import aiohttp
import logging
import json
import time
from asyncio import Semaphore
from typing import Dict, Any, List, Optional
from aiolimiter import AsyncLimiter
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    unique_id: str
    prompt: str
    response: Dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float
    error: Optional[str] = None
    request_id: Optional[str] = None
    model_version: Optional[str] = None
    estimated_cost: float = 0.0
    rate_limit_remaining: Optional[int] = None

class APIClient(ABC):
    def __init__(self, model_config: Dict[str, Any], rate_limit: int, dry_run: bool, model_metadata: List[Dict[str, Any]]):
        self.model_config = model_config
        self.dry_run = dry_run
        self.limiter = AsyncLimiter(rate_limit, 60)
        self.api_key = self._get_api_key()
        self.headers = self._get_headers()
        self.base_url = self._get_base_url()
        self.model_metadata = self._get_model_metadata(model_metadata)

    def _get_api_key(self) -> str:
        api_key_env_var = self.model_config['api_key']
        if api_key_env_var.startswith('${') and api_key_env_var.endswith('}'):
            api_key_env_var = api_key_env_var[2:-1]
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(f"API key environment variable '{api_key_env_var}' is not set")
        return api_key

    @abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def _get_base_url(self) -> str:
        pass

    @abstractmethod
    def _prepare_request_data(self, prompt: str, preload: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _get_api_endpoint(self) -> str:
        pass

    @abstractmethod
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _extract_token_info(self, response: Dict[str, Any]) -> Dict[str, int]:
        pass

    def _get_model_metadata(self, model_metadata: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        model_name = self.model_config['model_params'].get('model')
        for metadata in model_metadata:
            if metadata['model'] == model_name:
                return metadata
        logger.warning(f"No metadata found for model {model_name}")
        return None

    def _estimate_cost(self, token_info: Dict[str, int]) -> float:
        if not self.model_metadata:
            return 0.0

        input_cost = (token_info['prompt_tokens'] / 1_000_000) * self.model_metadata['cost_per_million_input_tokens']
        output_cost = (token_info['completion_tokens'] / 1_000_000) * self.model_metadata['cost_per_million_output_tokens']
        return input_cost + output_cost

    @abstractmethod
    async def send_request(self, session: aiohttp.ClientSession, prompt: str, preload: str, unique_id: str) -> APIResponse:
        pass

    @abstractmethod
    def extract_content(self, response: Dict[str, Any]) -> str:
        """Extract the content from the API response."""
        pass


class TogetherClient(APIClient):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _get_base_url(self) -> str:
        return self.model_config.get('base_url', 'https://api.together.xyz/v1')

    def _prepare_request_data(self, prompt: str, preload: str) -> Dict[str, Any]:
        return {
            "model": self.model_config['model_params'].get('model', 'mistralai/Mixtral-8x7B-Instruct-v0.1'),
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": preload}
            ],
            "max_tokens": self.model_config['model_params'].get('max_tokens', 1024),
            "temperature": self.model_config['model_params'].get('temperature', 0.7),
            "top_p": self.model_config['model_params'].get('top_p', 1),
            "top_k": self.model_config['model_params'].get('top_k', 0),
            "repetition_penalty": self.model_config['model_params'].get('repetition_penalty', 1),
            "stream": False,
            "logprobs": self.model_config['model_params'].get('logprobs', 0),
            "stop": self.model_config['model_params'].get('stop', None),
        }

    def _get_api_endpoint(self) -> str:
        return "/chat/completions"

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": response['id'],
            "content": [{"text": response['choices'][0]['message']['content']}],
            "model": response['model'],
            "usage": response['usage'],
            "created": response['created'],
            "object": response['object'],
            "choices": response['choices']
        }

    def _extract_token_info(self, response: Dict[str, Any]) -> Dict[str, int]:
        usage = response.get('usage', {})
        return {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0)
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=lambda retry_state: retry_state.outcome.exception() is not None
    )
    async def send_request(self, session: aiohttp.ClientSession, prompt: str, preload: str, unique_id: str) -> APIResponse:
        start_time = time.time()
        async with self.limiter:
            if self.dry_run:
                return APIResponse(
                    unique_id=unique_id,
                    prompt=prompt,
                    response={"dry_run": True, "message": "This is a dry run. No actual API call was made."},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency=0
                )

            data = self._prepare_request_data(prompt, preload)

            try:
                async with session.post(f"{self.base_url}{self._get_api_endpoint()}", json=data, headers=self.headers) as response:
                    response.raise_for_status()
                    json_response = await response.json()

                    parsed_response = self._parse_response(json_response)
                    token_info = self._extract_token_info(json_response)

                    return APIResponse(
                        unique_id=unique_id,
                        prompt=prompt,
                        response=parsed_response,
                        prompt_tokens=token_info['prompt_tokens'],
                        completion_tokens=token_info['completion_tokens'],
                        total_tokens=token_info['total_tokens'],
                        latency=time.time() - start_time,
                        request_id=json_response.get('id'),
                        model_version=json_response.get('model'),
                        estimated_cost=self._estimate_cost(token_info),
                        rate_limit_remaining=response.headers.get('X-RateLimit-Remaining')
                    )

            except aiohttp.ClientResponseError as e:
                logger.error(f"API error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                if e.status == 429:
                    logger.warning(f"Rate limit exceeded for {unique_id}. Retrying...")
                raise e
            except aiohttp.ClientError as e:
                logger.error(f"Network error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout error in request {unique_id}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except Exception as e:
                logger.error(f"Unexpected error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e

    def extract_content(self, response: Dict[str, Any]) -> str:
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')


class AnthropicClient(APIClient):

    def _get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }

    def _get_base_url(self) -> str:
        return self.model_config.get('base_url', 'https://api.anthropic.com')

    def _prepare_request_data(self, prompt: str, preload: str) -> Dict[str, Any]:
        return {
            "model": self.model_config['model_params'].get('model', 'claude-3-opus-20240307'),
            "max_tokens": self.model_config['model_params'].get('max_tokens', 1024),
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": preload}
            ],
            "temperature": self.model_config['model_params'].get('temperature', 1),
        }

    def _get_api_endpoint(self) -> str:
        return "/v1/messages"

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return response

    def _extract_token_info(self, response: Dict[str, Any]) -> Dict[str, int]:
        usage = response.get('usage', {})
        return {
            'prompt_tokens': usage.get('input_tokens', 0),
            'completion_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=lambda retry_state: retry_state.outcome.exception() is not None
    )
    async def send_request(self, session: aiohttp.ClientSession, prompt: str, preload: str, unique_id: str) -> APIResponse:
        start_time = time.time()
        async with self.limiter:
            if self.dry_run:
                return APIResponse(
                    unique_id=unique_id,
                    prompt=prompt,
                    response={"dry_run": True, "message": "This is a dry run. No actual API call was made."},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency=0
                )

            data = self._prepare_request_data(prompt, preload)

            try:
                async with session.post(f"{self.base_url}{self._get_api_endpoint()}", json=data, headers=self.headers) as response:
                    if response.status == 429:
                        rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                        if rate_limit_remaining == 0:
                            retry_after = int(response.headers.get('Retry-After', 0))
                            logger.warning(f"Rate limit exceeded for {unique_id}. Retrying after {retry_after} seconds.")
                            await asyncio.sleep(retry_after)
                            raise aiohttp.ClientResponseError(response.request_info, response.history, status=429,
                                                            message="Rate limit exceeded", headers=response.headers)
                    response.raise_for_status()
                    json_response = await response.json()

                    parsed_response = self._parse_response(json_response)
                    token_info = self._extract_token_info(json_response)

                    return APIResponse(
                        unique_id=unique_id,
                        prompt=prompt,
                        response=parsed_response,
                        prompt_tokens=token_info['prompt_tokens'],
                        completion_tokens=token_info['completion_tokens'],
                        total_tokens=token_info['total_tokens'],
                        latency=time.time() - start_time,
                        request_id=json_response.get('id'),
                        model_version=json_response.get('model'),
                        estimated_cost=self._estimate_cost(token_info),
                        rate_limit_remaining=response.headers.get('X-RateLimit-Remaining')
                    )

            except aiohttp.ClientResponseError as e:
                logger.error(f"API error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                if e.status == 429:
                    logger.warning(f"Rate limit exceeded for {unique_id}. Retrying...")
                raise e
            except aiohttp.ClientError as e:
                logger.error(f"Network error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout error in request {unique_id}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except Exception as e:
                logger.error(f"Unexpected error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e

    def extract_content(self, response: Dict[str, Any]) -> str:
        return response.get('content', [{}])[0].get('text', '')


class DeepseekClient(APIClient):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def _get_base_url(self) -> str:
        return self.model_config.get('base_url', 'https://api.deepseek.com')

    def _prepare_request_data(self, prompt: str, preload: str) -> Dict[str, Any]:
        return {
            "model": self.model_config['model_params'].get('model', 'deepseek-coder'),
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": preload}
            ],
            "max_tokens": self.model_config['model_params'].get('max_tokens', 2048),
            "temperature": self.model_config['model_params'].get('temperature', 1),
            "frequency_penalty": self.model_config['model_params'].get('frequency_penalty', 0),
            "presence_penalty": self.model_config['model_params'].get('presence_penalty', 0),
            "top_p": self.model_config['model_params'].get('top_p', 1),
            "stop": self.model_config['model_params'].get('stop', None),
            "stream": False,
            "logprobs": self.model_config['model_params'].get('logprobs', False),
            "top_logprobs": self.model_config['model_params'].get('top_logprobs', 0)
        }

    def _get_api_endpoint(self) -> str:
        return "/chat/completions"

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": response['id'],
            "content": [{"text": response['choices'][0]['message']['content']}],
            "model": response['model'],
            "usage": response['usage'],
            "created": response['created'],
            "object": response['object'],
            "choices": response['choices']
        }

    def _extract_token_info(self, response: Dict[str, Any]) -> Dict[str, int]:
        usage = response.get('usage', {})
        return {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0)
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=lambda retry_state: retry_state.outcome.exception() is not None
    )
    async def send_request(self, session: aiohttp.ClientSession, prompt: str, preload: str, unique_id: str) -> APIResponse:
        start_time = time.time()
        async with self.limiter:
            if self.dry_run:
                return APIResponse(
                    unique_id=unique_id,
                    prompt=prompt,
                    response={"dry_run": True, "message": "This is a dry run. No actual API call was made."},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency=0
                )

            data = self._prepare_request_data(prompt, preload)

            try:
                async with session.post(f"{self.base_url}{self._get_api_endpoint()}", json=data, headers=self.headers) as response:
                    if response.status == 429:
                        rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                        if rate_limit_remaining == 0:
                            retry_after = int(response.headers.get('Retry-After', 0))
                            logger.warning(f"Rate limit exceeded for {unique_id}. Retrying after {retry_after} seconds.")
                            await asyncio.sleep(retry_after)
                            raise aiohttp.ClientResponseError(response.request_info, response.history, status=429,
                                                            message="Rate limit exceeded", headers=response.headers)
                    response.raise_for_status()
                    json_response = await response.json()

                    parsed_response = self._parse_response(json_response)
                    token_info = self._extract_token_info(json_response)

                    return APIResponse(
                        unique_id=unique_id,
                        prompt=prompt,
                        response=parsed_response,
                        prompt_tokens=token_info['prompt_tokens'],
                        completion_tokens=token_info['completion_tokens'],
                        total_tokens=token_info['total_tokens'],
                        latency=time.time() - start_time,
                        request_id=json_response.get('id'),
                        model_version=json_response.get('model'),
                        estimated_cost=self._estimate_cost(token_info),
                        rate_limit_remaining=response.headers.get('X-RateLimit-Remaining')
                    )

            except aiohttp.ClientResponseError as e:
                logger.error(f"API error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                if e.status == 429:
                    logger.warning(f"Rate limit exceeded for {unique_id}. Retrying...")
                raise e
            except aiohttp.ClientError as e:
                logger.error(f"Network error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout error in request {unique_id}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except Exception as e:
                logger.error(f"Unexpected error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e

    def extract_content(self, response: Dict[str, Any]) -> str:
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')


class GroqClient(APIClient):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _get_base_url(self) -> str:
        return self.model_config.get('base_url', 'https://api.groq.com/openai/v1')

    def _prepare_request_data(self, prompt: str, preload: str) -> Dict[str, Any]:
        return {
            "model": self.model_config['model_params'].get('model', 'llama3-8b-8192'),
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": preload}
            ],
            "max_tokens": self.model_config['model_params'].get('max_tokens', 2048),
            "temperature": self.model_config['model_params'].get('temperature', 1),
        }

    def _get_api_endpoint(self) -> str:
        return "/chat/completions"

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": response['id'],
            "content": [{"text": response['choices'][0]['message']['content']}],
            "model": response['model'],
            "usage": response['usage'],
            "created": response['created'],
            "object": response['object'],
            "choices": response['choices'],
            "system_fingerprint": response.get('system_fingerprint'),
            "x_groq": response.get('x_groq')
        }

    def _extract_token_info(self, response: Dict[str, Any]) -> Dict[str, int]:
        usage = response.get('usage', {})
        return {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0)
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=lambda retry_state: retry_state.outcome.exception() is not None
    )
    async def send_request(self, session: aiohttp.ClientSession, prompt: str, preload: str, unique_id: str) -> APIResponse:
        start_time = time.time()
        async with self.limiter:
            if self.dry_run:
                return APIResponse(
                    unique_id=unique_id,
                    prompt=prompt,
                    response={"dry_run": True, "message": "This is a dry run. No actual API call was made."},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency=0
                )

            data = self._prepare_request_data(prompt, preload)

            try:
                async with session.post(f"{self.base_url}{self._get_api_endpoint()}", json=data, headers=self.headers) as response:
                    if response.status == 429:
                        rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                        if rate_limit_remaining == 0:
                            retry_after = int(response.headers.get('Retry-After', 0))
                            logger.warning(f"Rate limit exceeded for {unique_id}. Retrying after {retry_after} seconds.")
                            await asyncio.sleep(retry_after)
                            raise aiohttp.ClientResponseError(response.request_info, response.history, status=429,
                                                            message="Rate limit exceeded", headers=response.headers)
                    response.raise_for_status()
                    json_response = await response.json()

                    parsed_response = self._parse_response(json_response)
                    token_info = self._extract_token_info(json_response)

                    return APIResponse(
                        unique_id=unique_id,
                        prompt=prompt,
                        response=parsed_response,
                        prompt_tokens=token_info['prompt_tokens'],
                        completion_tokens=token_info['completion_tokens'],
                        total_tokens=token_info['total_tokens'],
                        latency=time.time() - start_time,
                        request_id=json_response.get('id'),
                        model_version=json_response.get('model'),
                        estimated_cost=self._estimate_cost(token_info),
                        rate_limit_remaining=response.headers.get('X-RateLimit-Remaining')
                    )

            except aiohttp.ClientResponseError as e:
                logger.error(f"API error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                if e.status == 429:
                    logger.warning(f"Rate limit exceeded for {unique_id}. Retrying...")
                raise e
            except aiohttp.ClientError as e:
                logger.error(f"Network error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout error in request {unique_id}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except Exception as e:
                logger.error(f"Unexpected error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e

    def extract_content(self, response: Dict[str, Any]) -> str:
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')


class OpenAIClient(APIClient):
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _get_base_url(self) -> str:
        return self.model_config.get('base_url', 'https://api.openai.com/v1')

    def _prepare_request_data(self, prompt: str, preload: str) -> Dict[str, Any]:
        return {
            "model": self.model_config['model_params'].get('model', 'gpt-3.5-turbo'),
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": preload}
            ],
            "temperature": self.model_config['model_params'].get('temperature', 0.7),
            "max_tokens": self.model_config['model_params'].get('max_tokens', 1024),
        }

    def _get_api_endpoint(self) -> str:
        return "/chat/completions"

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": response['id'],
            "content": [{"text": response['choices'][0]['message']['content']}],
            "model": response['model'],
            "usage": response['usage'],
            "created": response['created'],
            "object": response['object'],
            "choices": response['choices']
        }

    def _extract_token_info(self, response: Dict[str, Any]) -> Dict[str, int]:
        usage = response.get('usage', {})
        return {
            'prompt_tokens': usage.get('prompt_tokens', 0),
            'completion_tokens': usage.get('completion_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0)
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=lambda retry_state: retry_state.outcome.exception() is not None
    )
    async def send_request(self, session: aiohttp.ClientSession, prompt: str, preload: str, unique_id: str) -> APIResponse:
        start_time = time.time()
        async with self.limiter:
            if self.dry_run:
                return APIResponse(
                    unique_id=unique_id,
                    prompt=prompt,
                    response={"dry_run": True, "message": "This is a dry run. No actual API call was made."},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency=0
                )

            data = self._prepare_request_data(prompt, preload)

            try:
                async with session.post(f"{self.base_url}{self._get_api_endpoint()}", json=data, headers=self.headers) as response:
                    if response.status == 429:
                        rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                        if rate_limit_remaining == 0:
                            retry_after = int(response.headers.get('Retry-After', 60))
                            logger.warning(f"Rate limit exceeded for {unique_id}. Retrying after {retry_after} seconds.")
                            await asyncio.sleep(retry_after)
                            raise aiohttp.ClientResponseError(response.request_info, response.history, status=429,
                                                            message="Rate limit exceeded", headers=response.headers)
                    response.raise_for_status()
                    json_response = await response.json()

                    parsed_response = self._parse_response(json_response)
                    token_info = self._extract_token_info(json_response)

                    return APIResponse(
                        unique_id=unique_id,
                        prompt=prompt,
                        response=parsed_response,
                        prompt_tokens=token_info['prompt_tokens'],
                        completion_tokens=token_info['completion_tokens'],
                        total_tokens=token_info['total_tokens'],
                        latency=time.time() - start_time,
                        request_id=json_response.get('id'),
                        model_version=json_response.get('model'),
                        estimated_cost=self._estimate_cost(token_info),
                        rate_limit_remaining=response.headers.get('X-RateLimit-Remaining')
                    )

            except aiohttp.ClientResponseError as e:
                logger.error(f"API error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                if e.status == 429:
                    logger.warning(f"Rate limit exceeded for {unique_id}. Retrying...")
                raise e
            except aiohttp.ClientError as e:
                logger.error(f"Network error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout error in request {unique_id}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except Exception as e:
                logger.error(f"Unexpected error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e

    def extract_content(self, response: Dict[str, Any]) -> str:
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')


class ReplicateClient(APIClient):
    def __init__(self, model_config: Dict[str, Any], rate_limit: int, dry_run: bool, model_metadata: List[Dict[str, Any]]):
        super().__init__(model_config, rate_limit, dry_run, model_metadata)
        self.model = self.model_config['model_params'].get('model')
        if not self.model:
            raise ValueError("model must be specified in model_params for Replicate models")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _get_base_url(self) -> str:
        return self.model_config.get('base_url', 'https://api.replicate.com/v1')

    def _prepare_request_data(self, prompt: str, preload: str) -> Dict[str, Any]:
        return {
            "input": {
                "top_k": 0,
                "top_p": self.model_config['model_params'].get('top_p', 0.9),
                "prompt": prompt,
                "max_tokens": self.model_config['model_params'].get('max_tokens', 512),
                "min_tokens": 0,
                "temperature": self.model_config['model_params'].get('temperature', 0.6),
                "system_prompt": "You are a helpful assistant",
                "length_penalty": 1,
                "stop_sequences": "<|end_of_text|>,<|eot_id|>",
                "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "presence_penalty": 1.15,
                "frequency_penalty": 0.2,
                "log_performance_metrics": False
            }
        }

    def _get_api_endpoint(self) -> str:
        return f"/models/{self.model}/predictions"

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": response['id'],
            "content": [{"text": ''.join(response['output']) if response['output'] else ""}],
            "model": self.model,
            "created": response['created_at'],
            "status": response['status'],
            "error": response['error'],
            "metrics": response.get('metrics', {})
        }

    def _extract_token_info(self, response: Dict[str, Any]) -> Dict[str, int]:
        metrics = response.get('metrics', {})
        return {
            'prompt_tokens': metrics.get('input_token_count', 0),
            'completion_tokens': metrics.get('output_token_count', 0),
            'total_tokens': metrics.get('input_token_count', 0) + metrics.get('output_token_count', 0)
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=lambda retry_state: retry_state.outcome.exception() is not None
    )
    async def send_request(self, session: aiohttp.ClientSession, prompt: str, preload: str, unique_id: str) -> APIResponse:
        start_time = time.time()
        async with self.limiter:
            if self.dry_run:
                return APIResponse(
                    unique_id=unique_id,
                    prompt=prompt,
                    response={"dry_run": True, "message": "This is a dry run. No actual API call was made."},
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency=0
                )

            data = self._prepare_request_data(prompt, preload)

            try:
                async with session.post(f"{self.base_url}{self._get_api_endpoint()}", json=data, headers=self.headers) as response:
                    response.raise_for_status()
                    json_response = await response.json()

                    # Replicate API returns a status, we need to poll until it's completed
                    while json_response['status'] not in ['succeeded', 'failed', 'canceled']:
                        await asyncio.sleep(1)
                        async with session.get(json_response['urls']['get'], headers=self.headers) as status_response:
                            status_response.raise_for_status()
                            json_response = await status_response.json()

                    parsed_response = self._parse_response(json_response)
                    token_info = self._extract_token_info(json_response)

                    return APIResponse(
                        unique_id=unique_id,
                        prompt=prompt,
                        response=parsed_response,
                        prompt_tokens=token_info['prompt_tokens'],
                        completion_tokens=token_info['completion_tokens'],
                        total_tokens=token_info['total_tokens'],
                        latency=time.time() - start_time,
                        request_id=json_response.get('id'),
                        model_version=json_response.get('version'),
                        estimated_cost=self._estimate_cost(token_info),
                        rate_limit_remaining=None  # Replicate API doesn't provide this information
                    )

            except aiohttp.ClientResponseError as e:
                logger.error(f"API error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except aiohttp.ClientError as e:
                logger.error(f"Network error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout error in request {unique_id}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e
            except Exception as e:
                logger.error(f"Unexpected error in request {unique_id}: {str(e)}")
                logger.debug(f"Payload for request {unique_id}: {json.dumps(data, indent=2)}")
                raise e

    def extract_content(self, response: Dict[str, Any]) -> str:
        return response.get('content', [{}])[0].get('text', '')


def get_api_client(model_config: Dict[str, Any], rate_limit: int, dry_run: bool, model_metadata: List[Dict[str, Any]]) -> APIClient:
    provider = model_config['provider']
    client_classes = {
        "anthropic": AnthropicClient,
        "deepseek": DeepseekClient,
        "groq": GroqClient,
        "openai": OpenAIClient,
        "replicate": ReplicateClient,
        "together": TogetherClient
    }
    client_class = client_classes.get(provider)
    if client_class:
        return client_class(model_config, rate_limit, dry_run, model_metadata)
    else:
        raise ValueError(f"Unsupported API provider: {provider}")
