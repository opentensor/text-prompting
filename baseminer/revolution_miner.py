# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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

import os
import copy
import time
import wandb
import argparse
import pydantic
import threading
import traceback

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union

import bittensor as bt
from prompting.protocol import Prompting

from baseminer.priority import priority
from baseminer.blacklist import blacklist
from baseminer.run import run
from baseminer.set_weights import set_weights
from baseminer.config import check_config, get_config



class Miner(ABC):
    """
    The Miner class is an abstract base class that defines the structure for Bittensor miners.
    Subclassed should implement the `prompt` method to define their own response logic.
    The `blacklist` and `priority` methods can also be overridden to provide custom logic.
    """

    def __init__(self, config=None, axon=None, wallet=None, subtensor=None):
        """
        Initializes the Miner with the given configurations and Bittensor objects.

        Args:
            config: Configuration object that holds settings for the miner.
            axon: Bittensor Axon object which handles incoming requests.
            wallet: Bittensor Wallet object which holds cryptographic keys.
            subtensor: Bittensor Subtensor object which manages the blockchain connection.
        """
        # Setup base config from Miner.config() and merge with subclassed config.
        base_config = copy.deepcopy(config or get_config())
        self.config = self.config()
        self.config.merge(base_config)

        check_config(Miner, self.config)
        bt.logging.info(self.config)  # TODO: duplicate print?

        self.prompt_cache: Dict[str, Tuple[str, int]] = {}

        # Activating Bittensor's logging with the set configurations.
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(
            f"Running miner for subnet: {self.config.netuid} on network: {self.config.subtensor.chain_endpoint} with config:"
        )

        if not self.config.miner.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.miner.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk. "
            )

        bt.logging.info("Setting up bittensor objects.")

        # Wallet holds cryptographic information, ensuring secure transactions and communication.
        self.wallet = wallet or bt.wallet(config=self.config)
        bt.logging.info(f"Wallet {self.wallet}")

        # subtensor manages the blockchain connection, facilitating interaction with the Bittensor blockchain.
        self.subtensor = subtensor or bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {subtensor}")

        # metagraph provides the network's current state, holding state about other participants in a subnet.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(
                f"\nYour validator: {self.wallet} if not registered to chain connection: {self.subtensor} \nRun btcli register and try again. "
            )
            exit()
        else:
            # Each miner gets a unique identity (UID) in the network for differentiation.
            self.my_subnet_uid = self.metagraph.hotkeys.index(
                self.wallet.hotkey.ss58_address
            )
            bt.logging.info(f"Running miner on uid: {self.my_subnet_uid}")

        # The axon handles request processing, allowing validators to send this process requests.
        self.axon = axon or bt.axon(wallet=self.wallet, port=self.config.axon.port)
        bt.logging.info(f"Axon {self.axon}")

        if self.config.wandb.on:
            tags = [self.wallet.hotkey.ss58_address, f"netuid_{self.config.netuid}"]
            self.wandb_run = wandb.init(
                project=self.config.wandb.project_name,
                entity=self.config.wandb.entity,
                config=self.config,
                mode="online" if self.config.wandb.on else "offline",
                dir=self.config.miner.full_path,
                magic=True,
                tags=tags,
            )

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.thread: threading.Thread = None

        self.request_timestamps: Dict = {}

    @abstractmethod
    def config(self) -> "bt.Config":
        ...

    @classmethod
    @abstractmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        ...

    @abstractmethod
    def prompt(self, synapse: Prompting) -> Prompting:
        ...

    def blacklist(self, synapse: Prompting) -> Tuple[bool, str]:
        """
        Default blacklist logic

        Define how miners should blacklist requests. This Function
        Runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Below: Check that the hotkey is a registered entity in the metagraph.

        Args:
            synapse (:obj:`bittensor.synapse.Synapse`, `required`):
                synapse object containing the request headers.
        Returns:
            blacklisted (:obj:`bool`):
        """

        def _blacklist(synapse: "Prompting") -> Tuple[bool, str]:
            raise NotImplementedError("blacklist not implemented in subclass")

        return blacklist(self, _blacklist, synapse)

    def priority(self, synapse: Prompting) -> float:
        """
        Define how miners should prioritize requests.

        Miners may recieve messages from multiple entities at once. This function
        determines which request should be processed first. Higher values indicate
        that the request should be processed first. Lower values indicate that the
        request should be processed later.

        Below: simple logic, prioritize requests from entities with more stake.

        Args:
            synapse (:obj:`bittensor.synapse.Synapse`, `required`):
                synapse object containing the request headers.
        Returns:
            priority (:obj:`float`):
        """

        def _priority(synapse: "Prompting") -> bool:
            raise NotImplementedError("priority not implemented in subclass")

        return priority(self, _priority, synapse)

    def run(self):
        """
        Runs the miner logic. This method starts the miner's operations, including
        listening for incoming requests and periodically updating the miner's knowledge
        of the network graph.
        """
        run(self)

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.thread.join(5)
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.

        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      None if the context was exited without an exception.
            exc_value: The instance of the exception that caused the context to be exited.
                       None if the context was exited without an exception.
            traceback: A traceback object encoding the stack trace.
                       None if the context was exited without an exception.
        """
        self.stop_run_thread()
