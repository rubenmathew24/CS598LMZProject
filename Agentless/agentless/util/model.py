import json
from abc import ABC, abstractmethod
from typing import List

from agentless.util.api_requests import (
    create_anthropic_config,
    create_chatgpt_config,
    request_anthropic_engine,
    request_chatgpt_engine,
)

from google import genai
from google.genai import types
import os
import asyncio


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        logger,
        batch_size: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 1024,
    ) -> None:
        logger.info("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.logger = logger
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1
        batch_size = min(self.batch_size, num_samples)

        config = create_chatgpt_config(
            message=message,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            batch_size=batch_size,
            model=self.name,
        )
        ret = request_chatgpt_engine(config, self.logger)
        if ret:
            responses = [choice.message.content for choice in ret.choices]
            completion_tokens = ret.usage.completion_tokens
            prompt_tokens = ret.usage.prompt_tokens
        else:
            responses = [""]
            completion_tokens = 0
            prompt_tokens = 0

        # The nice thing is, when we generate multiple samples from the same input (message),
        # the input tokens are only charged once according to openai API.
        # Therefore, we assume the request cost is only counted for the first sample.
        # More specifically, the `prompt_tokens` is for one input message,
        # and the `completion_tokens` is the sum of all returned completions.
        # Therefore, for the second and later samples, the cost is zero.
        trajs = [
            {
                "response": responses[0],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            }
        ]
        for response in responses[1:]:
            trajs.append(
                {
                    "response": response,
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                    },
                }
            )
        return trajs

    def is_direct_completion(self) -> bool:
        return False


class AnthropicChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    _STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for editing files
* State is persistent across command calls and discussions with the user

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

    _USER_REPLY_EDIT_MESSAGE = """File is successfully edited"""

    tools = [
        {
            "name": "str_replace_editor",
            "description": _STR_REPLACE_EDITOR_DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "description": "Full path to file, e.g. `folder/file.py`.",
                        "type": "string",
                    },
                    "old_str": {
                        "description": "Required parameter containing the string in `path` to replace.",
                        "type": "string",
                    },
                    "new_str": {
                        "description": "Optional parameter containing the new string (if not given, no string will be added).",
                        "type": "string",
                    },
                },
                "required": ["path", "old_str"],
            },
        }
    ]

    MAX_CODEGEN_ITERATIONS = 10

    # specialized codegen with tool
    def codegen_w_tool(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        def _build_response_and_extract(response, messages, iter):
            json_response = response.to_dict()

            contains_tool = False
            # formulate the messages
            json_response.pop("id")
            json_response.pop("model")
            json_response.pop("stop_reason")
            json_response.pop("stop_sequence")
            json_response.pop("type")
            json_response.pop("usage")

            messages.append(json_response)

            response_content = []

            for json_message in json_response["content"]:
                if (json_message["type"] == "tool_use"):
                    contains_tool = True
                    # each tool use requires a response
                    response_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": json_message["id"],
                            "content": self._USER_REPLY_EDIT_MESSAGE,
                        }
                    )

            if contains_tool:
                messages.append(
                    {
                        "role": "user",
                        "content": response_content,
                    }
                )
            else:
                if iter == 0:
                    # if the first iteration does not contain the tool, likely the model is doing some CoT for debugging
                    # append encouraging message
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Please generate editing commands to fix the issue",
                                }
                            ],
                        }
                    )
                    contains_tool = True

            return messages, contains_tool

        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            self.logger.info(f" === Generating ====")
            # initialized the traj
            traj = {
                "response": [],
                "usage": {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "cache_creation_token": 0,
                    "cache_read_input_tokens": 0,
                },
            }

            # create the initial config and messages
            messages = [
                {"role": "user", "content": [{"type": "text", "text": message}]}
            ]

            for iteration in range(self.MAX_CODEGEN_ITERATIONS):
                config = create_anthropic_config(
                    message=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    batch_size=1,
                    model=self.name,
                    tools=self.tools,
                )
                ret = request_anthropic_engine(
                    config,
                    self.logger,
                    prompt_cache=True,  # prompt cache should be always true as we at least should query twice
                )

                if ret:
                    # add the response to the traj
                    traj["response"].append([reply.to_dict() for reply in ret.content])

                    # pretty dump the response
                    for reply in ret.content:
                        self.logger.info(json.dumps(reply.to_dict(), indent=2))

                    # update the usage
                    traj["usage"]["completion_tokens"] += ret.usage.output_tokens
                    traj["usage"]["prompt_tokens"] += ret.usage.input_tokens
                    traj["usage"][
                        "cache_creation_token"
                    ] += ret.usage.cache_creation_input_tokens
                    traj["usage"][
                        "cache_read_input_tokens"
                    ] += ret.usage.cache_read_input_tokens

                    messages, contains_tool = _build_response_and_extract(
                        ret, messages, iteration
                    )

                    if not contains_tool:
                        break
                else:
                    assert (
                        False
                    ), "No response from the engine"  # this should not happen

            if ret:
                trajs.append(traj)
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_anthropic_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_anthropic_engine(
                config, self.logger, prompt_cache=prompt_cache
            )

            if ret:
                trajs.append(
                    {
                        "response": ret.content[0].text,
                        "usage": {
                            "completion_tokens": ret.usage.output_tokens,
                            "prompt_tokens": ret.usage.input_tokens,
                            "cache_creation_token": 0
                            if not prompt_cache
                            else ret.usage.cache_creation_input_tokens,
                            "cache_read_input_tokens": 0
                            if not prompt_cache
                            else ret.usage.cache_read_input_tokens,
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class DeepSeekChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)

    def codegen(
        self, message: str, num_samples: int = 1, prompt_cache: bool = False
    ) -> List[dict]:
        if self.temperature == 0:
            assert num_samples == 1

        trajs = []
        for _ in range(num_samples):
            config = create_chatgpt_config(
                message=message,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                batch_size=1,
                model=self.name,
            )
            ret = request_chatgpt_engine(
                config, self.logger, base_url="https://api.deepseek.com"
            )
            if ret:
                trajs.append(
                    {
                        "response": ret.choices[0].message.content,
                        "usage": {
                            "completion_tokens": ret.usage.completion_tokens,
                            "prompt_tokens": ret.usage.prompt_tokens,
                        },
                    }
                )
            else:
                trajs.append(
                    {
                        "response": "",
                        "usage": {
                            "completion_tokens": 0,
                            "prompt_tokens": 0,
                        },
                    }
                )

        return trajs

    def is_direct_completion(self) -> bool:
        return False


class GeminiChatDecoder(DecoderBase):
    def __init__(self, name: str, logger, **kwargs) -> None:
        super().__init__(name, logger, **kwargs)
        self.logger.info(f"Initializing Gemini decoder with model: {self.name}")
        
        # Retrieve your Gemini API key from the environment
        self.api_key = os.environ.get("GEMINI_API_KEY")
        
        # Create the GenAI client
        self.client = genai.Client(api_key=self.api_key)
        
        # Map the user-facing model name to the actual Gemini model ID if needed
        if self.name == "gemini-2.5":
            self.real_model = "gemini-2.0-flash-lite"
        else:
            self.real_model = self.name

    def _send_message_safely(self, message):
        """Thread-safe wrapper for sending messages to Gemini API"""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create a chat session
            chat = self.client.aio.chats.create(
                model=self.real_model,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                ),
            )
            
            try:
                # Run the async call in this new loop
                response = loop.run_until_complete(chat.send_message([message]))
                return response
            finally:
                # Clean up
                loop.close()
        except Exception as e:
            self.logger.error(f"Error in Gemini API call: {str(e)}")
            return None

    def codegen(self, message: str, num_samples: int = 1, prompt_cache: bool = False) -> List[dict]:
        """
        Generates content using Gemini and returns a list of trajectories.
        Each trajectory is a dict with keys "response" and "usage".
        """
        if self.temperature == 0:
            assert num_samples == 1

        self.logger.info(f"Gemini codegen: {num_samples} sample(s) at temperature {self.temperature}")

        # Collect responses from Gemini
        responses = []
        for i in range(num_samples):
            self.logger.info(f"Gemini generation {i+1}/{num_samples}")
            
            # Use thread-safe wrapper
            reply = self._send_message_safely(message)
            
            if reply and hasattr(reply, 'text'):
                responses.append(reply.text)
            else:
                self.logger.warning("Failed to get valid response from Gemini")
                responses.append("")

        # Build trajectory list similar to other decoders
        trajs = []
        if responses:
            # First sample includes estimated token usage
            if responses[0]:
                # Estimate token counts (approx 1.3 tokens per word)
                input_words = len(message.split())
                output_words = len(responses[0].split())
                prompt_tokens = max(1, int(input_words * 1.3))
                completion_tokens = max(1, int(output_words * 1.3))
            else:
                prompt_tokens = 0
                completion_tokens = 0
                
            trajs.append({
                "response": responses[0],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                },
            })
            
            # Subsequent samples have zero usage (assuming cost is charged only once)
            for resp in responses[1:]:
                trajs.append({
                    "response": resp,
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                    },
                })
        else:
            # Return empty response if all attempts failed
            trajs.append({
                "response": "",
                "usage": {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                },
            })
            
        return trajs

    def is_direct_completion(self) -> bool:
        return False


def make_model(
    model: str,
    backend: str,
    logger,
    batch_size: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.0,
):
    if backend == "openai":
        return OpenAIChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "anthropic":
        return AnthropicChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "deepseek":
        return DeepSeekChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    elif backend == "gemini":
        return GeminiChatDecoder(
            name=model,
            logger=logger,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise NotImplementedError
