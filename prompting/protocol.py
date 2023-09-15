# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pydantic
import time
import torch
from typing import List
import bittensor as bt


class Prompting(bt.Synapse):
    """
    The Prompting subclass of the Synapse class encapsulates the functionalities related to prompting scenarios.

    It specifies three fields - `roles`, `messages` and `completion` - that define the state of the Prompting object.
    The `roles` and `messages` are read-only fields defined during object initialization, and `completion` is a mutable
    field that can be updated as the prompting scenario progresses.

    The Config inner class specifies that assignment validation should occur on this class (validate_assignment = True),
    meaning value assignments to the instance fields are checked against their defined types for correctness.

    Attributes:
        roles (List[str]): A list of roles in the prompting scenario. This field is both mandatory and immutable.
        messages (List[str]): A list of messages in the prompting scenario. This field is both mandatory and immutable.
        completion (str): A string that captures completion of the prompt. This field is mutable.

    Methods:
        deserialize() -> "Prompting": Returns the instance of the current object.


    The `Prompting` class also overrides the `deserialize` method, returning the
    instance itself when this method is invoked. Additionally, it provides a `Config`
    inner class that enforces the validation of assignments (`validate_assignment = True`).

    Here is an example of how the `Prompting` class can be used:

    ```python
    # Create a Prompting instance
    prompt = Prompting(roles=["system", "user"], messages=["Hello", "Hi"])

    # Print the roles and messages
    print("Roles:", prompt.roles)
    print("Messages:", prompt.messages)

    # Update the completion
    model_prompt =... # Use prompt.roles and prompt.messages to generate a prompt
    for your LLM as a single string.
    prompt.completion = model(model_prompt)

    # Print the completion
    print("Completion:", prompt.completion)
    ```

    This will output:
    ```
    Roles: ['system', 'user']
    Messages: ['You are a helpful assistant.', 'Hi, what is the meaning of life?']
    Completion: "The meaning of life is 42. Deal with it, human."
    ```

    This example demonstrates how to create an instance of the `Prompting` class, access the
    `roles` and `messages` fields, and update the `completion` field.


    """

    class Config:
        """
        Pydantic model configuration class for Prompting. This class sets validation of attribute assignment as True.
        validate_assignment set to True means the pydantic model will validate attribute assignments on the class.
        """

        validate_assignment = True

    def deserialize(self) -> "Prompting":
        """
        Returns the instance of the current Prompting object.

        This method is intended to be potentially overridden by subclasses for custom deserialization logic.
        In the context of the Prompting class, it simply returns the instance itself. However, for subclasses
        inheriting from this class, it might give a custom implementation for deserialization if need be.

        Returns:
            Prompting: The current instance of the Prompting class.
        """
        return self

    roles: List[str] = pydantic.Field(
        ...,
        title="Roles",
        description="A list of roles in the Prompting scenario. Immuatable.",
        allow_mutation=False,
    )

    messages: List[str] = pydantic.Field(
        ...,
        title="Messages",
        description="A list of messages in the Prompting scenario. Immutable.",
        allow_mutation=False,
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current Prompting object. This attribute is mutable and can be updated.",
    )


class StreamPrompting(bt.StreamingSynapse):
    """
    StreamPrompting is a subclass of StreamingSynapse that is specifically designed for prompting network functionality.
    It overrides abstract methods from the parent class to provide concrete implementations for processing streaming responses,
    deserializing the response, and extracting JSON data.

    Attributes:
        roles: List of roles associated with the prompt.
        messages: List of messages to be processed.
        completion: A string to store the completion result.
    """

    roles: List[str] = pydantic.Field(
        ...,
        title="Roles",
        description="A list of roles in the Prompting scenario. Immuatable.",
        allow_mutation=False,
    )

    messages: List[str] = pydantic.Field(
        ...,
        title="Messages",
        description="A list of messages in the Prompting scenario. Immutable.",
        allow_mutation=False,
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current Prompting object. This attribute is mutable and can be updated.",
    )

    async def process_streaming_response(self, response):
        """
        Processes the streaming response by iterating through the content and decoding tokens.
        Concatenates the decoded tokens into the completion attribute.

        Args:
            response: The response object containing the content to be processed.
        """
        if self.completion is None:
            self.completion = ""
        async for chunk in response.content.iter_any():
            tokens = chunk.decode('utf-8').split('\n')
            for token in tokens:
                if token:
                    self.completion += token

    def deserialize(self):
        """
        Deserializes the response by returning the completion attribute.

        Returns:
            str: The completion result.
        """
        return self.completion

    def extract_response_json(self, response):
        """
        Extracts JSON data from the response, including headers and specific information related to dendrite and axon.

        Args:
            response: The response object from which to extract JSON data.

        Returns:
            dict: A dictionary containing extracted JSON data.
        """
        headers = {k.decode('utf-8'): v.decode('utf-8') for k, v in response.__dict__["_raw_headers"]}

        def extract_info(prefix):
            return {key.split('_')[-1]: value for key, value in headers.items() if key.startswith(prefix)}

        return {
            "name": headers.get('name', ''),
            "timeout": float(headers.get('timeout', 0)),
            "total_size": int(headers.get('total_size', 0)),
            "header_size": int(headers.get('header_size', 0)),
            "dendrite": extract_info('bt_header_dendrite'),
            "axon": extract_info('bt_header_axon'),
            "roles": self.roles,
            "messages": self.messages,
            "completion": self.completion,
        }