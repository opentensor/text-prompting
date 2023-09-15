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

# Bittensor Miner Template:# Step 1: Import necessary libraries and modules
"""
Bittensor Miner Template

This is a basic template for creating a Bittensor miner. This miner listens to incoming requests 
from the Bittensor network and responds with a static message: "I am a chatbot".

Developers can extend this template by introducing more sophisticated response generation 
mechanisms, custom configurations, and additional functionalities.
"""

import time
import argparse
from prompting.baseminer.miner import Miner
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
        parser = argparse.ArgumentParser(description="OpenAI Miner Configs")
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
        Encapsulates the logic for generating a streaming response. It defines the tokenizer, model, and prompt functions,
        and creates a streaming response using the provided synapse.

        Args:
            synapse: A StreamPrompting instance containing the messages to be processed.

        Returns:
            StreamPrompting: The streaming response object.
        """
        def tokenizer(text):
            return (ord(char) for char in text)

        def model(ids):
            return (chr(id) for id in ids)

        async def prompt(text: str, send: Send):
            # Simulate model inference
            input_ids = tokenizer(text)
            for token in model(input_ids):
                await send({"type": "http.response.body", "body": (token + '\n').encode('utf-8'), "more_body": True})
                bt.logging.trace(f"Streamed token: {token}")
                # Sleep to show the streaming effect
                await asyncio.sleep(0.5)

        message = synapse.messages[0]
        token_streamer = partial(prompt, message)
        return synapse.create_streaming_response(token_streamer)


# This is the main function, which runs the miner.
if __name__ == "__main__":
    """
    Main execution point for the TemplateMiner.

    This script initializes and runs the TemplateMiner, connecting it to the Bittensor network.
    The miner listens for incoming requests and responds with a static message.

    Developers can start the miner by simply executing this script. To add more functionalities
    or customize the miner's behavior, consider extending the TemplateMiner class above.
    """
    with TemplateMiner():
        while True:
            print("running...", time.time())
            time.sleep(1)
