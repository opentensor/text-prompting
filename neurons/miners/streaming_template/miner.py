# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

# Bittensor Miner Template:# Step 1: Import necessary libraries and modules
"""
Bittensor Miner Template

This is a basic template for creating a Bittensor miner. This miner listens to incoming requests 
from the Bittensor network and responds with a static message: "I am a chatbot".

Developers can extend this template by introducing more sophisticated response generation 
mechanisms, custom configurations, and additional functionalities.
"""

import asyncio
import time
import argparse
from functools import partial
from starlette.types import Send
from prompting.baseminer.miner import Miner
from transformers import GPT2Tokenizer
from prompting.protocol import StreamPrompting
import bittensor as bt


class StreamingTemplateMiner(Miner):
    def config(self) -> "bt.Config":
        """
        Returns the configuration object specific to this miner.

        Implement and extend this method to provide custom configurations for the miner.
        Currently, it sets up a basic configuration parser.

        Returns:
            bt.Config: A configuration object with the miner's operational parameters.
        """
        parser = argparse.ArgumentParser(description="Streaming Miner Configs")
        self.add_args(parser)
        return bt.config(parser)

    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds custom arguments to the command line parser.

        Developers can introduce additional command-line arguments specific to the miner's
        functionality in this method. These arguments can then be used to configure the miner's operation.

        Args:
            parser (argparse.ArgumentParser):
                The command line argument parser to which custom arguments should be added.
        """
        pass

    def prompt(self, synapse: StreamPrompting) -> StreamPrompting:
        """
        Generates a streaming response for the provided synapse.

        This function serves as the main entry point for handling streaming prompts. It takes 
        the incoming synapse which contains messages to be processed and returns a streaming 
        response. The function uses the GPT-2 tokenizer and a simulated model to tokenize and decode 
        the incoming message, and then sends the response back to the client token by token.

        Args:
            synapse (StreamPrompting): The incoming StreamPrompting instance containing the messages to be processed.

        Returns:
            StreamPrompting: The streaming response object which can be used by other functions to 
                            stream back the response to the client.

        Usage:
            This function can be extended and customized based on specific requirements of the 
            miner. Developers can swap out the tokenizer, model, or adjust how streaming responses 
            are generated to suit their specific applications.
        """
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Simulated function to decode token IDs into strings. In a real-world scenario, 
        # this can be replaced with an actual model inference step.
        def model(ids):
            return (tokenizer.decode(id) for id in ids)

        async def _prompt(text: str, send: Send):
            """
            Asynchronously processes the input text and sends back tokens as a streaming response.

            This function takes an input text, tokenizes it using the GPT-2 tokenizer, and then 
            uses the simulated model to decode token IDs into strings. It then sends each token 
            back to the client as a streaming response, with a delay between tokens to simulate 
            the effect of real-time streaming.

            Args:
                text (str): The input text message to be processed.
                send (Send): An asynchronous function that allows sending back the streaming response.

            Usage:
                This function can be adjusted based on the streaming requirements, speed of 
                response, or the model being used. Developers can also introduce more sophisticated 
                processing steps or modify how tokens are sent back to the client.
            """
            # Simulate model inference
            input_ids = tokenizer(text, return_tensors="pt").input_ids.squeeze()
            # Iterate over the decoded tokens and send them back to the client.
            for token in model(input_ids):
                # Send token back to the client
                await send({"type": "http.response.body", "body": (token + '\n').encode('utf-8'), "more_body": True})
                bt.logging.trace(f"Streamed token: {token}")
                # Sleep to show the streaming effect
                await asyncio.sleep(1)

        message = synapse.messages[0]
        # Bind the input message to the _prompt function and return the streaming response.
        token_streamer = partial(_prompt, message)
        return synapse.create_streaming_response(token_streamer)


# This is the main function, which runs the miner.
if __name__ == "__main__":
    """
    Entry point for executing the StreamingTemplateMiner.

    This block initializes the StreamingTemplateMiner and runs it, effectively connecting 
    it to the Bittensor network. Once connected, the miner will continuously listen for 
    incoming requests from the Bittensor network. For every request, it responds with a 
    static message processed as per the logic defined in the 'prompt' method of the 
    StreamingTemplateMiner class.

    The main loop at the end serves to keep the miner running indefinitely. It periodically 
    prints a "running..." message to the console, providing a simple indication that the miner 
    is operational and active.

    Developers looking to extend or customize the miner's behavior can modify the 
    StreamingTemplateMiner class and its methods. However, this block itself usually 
    remains unchanged unless there's a need for specific startup behaviors or configurations.

    To start the miner:
    Simply execute this script. Ensure all dependencies are properly installed and network 
    configurations are correctly set up.
    """
    with StreamingTemplateMiner():
        while True:
            print("running...", time.time())
            time.sleep(1)
