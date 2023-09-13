
<div align="center">

# **Bittensor Subnet Template** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

### The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)

</div>

---

This template contains all the necessary files and functions to run Bittensor's Text-Prompting Subnet. You can try running miners on netuid 8 in Bittensor's test network.

# Introduction
The Bittensor blockchain hosts multiple self-contained incentive mechanisms 'subnets'. Subnets are playing fields through which miners (those producing value) and validators (those producing consensus) determine together the proper distribution of TAO for the purpose of incentivizing the creation of value, i.e. generating digital commodities, such as intelligence, or data. Each consists of a wire protocol through which miners and validators interact and their method of interacting with Bittensor's chain consensus engine [Yuma Consensus](https://bittensor.com/documentation/validating/yuma-consensus) which is designed to drive these actors into agreement about who is creating value.

This repository is a subnet for text prompting with large language models (LLM). Inside, you will find miners and validators designed by the OpenTensor Foundation team to valdiate and serve language models. The current validator implementation queries the network for responses while servers responds to requests with their best completions. These completions are judged and ranked by the validators and passed to the chain. 

</div>

---

# Installation
This repository requires python3.8 or higher. To install, simply clone this repository and install the requirements.
```bash
git clone https://github.com/opentensor/text-prompting.git
cd text-prompting
python -m pip install -r requirements.txt
python -m pip install -e .
```

If you are running a specific server, you might need install server-specific requirements.

```bash
cd neurons/miners/bittensorLM
python -m pip install -r requirements.txt
```

</div>

---

Once you have installed this repo, you can run the miner and validator with the following commands.
```bash
# To run the miner
python -m neurons/miners/bittensorLM/miner.py 
    --netuid 8  
    --subtensor.network test 
    --wallet.name <your miner wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode

# To run the validator
python -m neurons/validators/validator.py
    --netuid 8
    --subtensor.chain_endpoint test 
    --wallet.name <your validator wallet>  # Must be created using the bittensor-cli
    --wallet.hotkey <your validator hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
```

</div>

---

## License
This repository is licensed under the MIT License.
```text
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
```
