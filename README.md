# Micro-Price Trading Reinforcement Learning

---

## Background
This project began as a capstone project in my final semester of my MEng program in Financial Engineering in Fall
of 2021. Most of the work has currently been done as part of a team of four. Building off of the work done by Dr. Sasha
Stoikov in his paper [The Micro-Price: A High Frequency Estimator of Future Prices](https://deliverypdf.ssrn.com/delivery.php?ID=252085073120027109121001079004121086116042064082020028029003104121070087111072016028034036040047022047027073067018116065085084050076003080012026097004127111070006027057010020106084117017068006106093026122100005080089095064116088090123082115123123085100&EXT=pdf&INDEX=TRUE)
along with the [simulation framework](https://github.com/xhshenxin/Micro_Price) created by a previous group of students,
our main focus was on creating a reinforcement learning environment, primarily for pairs-trading, and attempting to
improve the simulation model. Near the end of the semester, our project sponsor urged us to change direction and pursue
an optimal execution algorithm instead, explaining both environments found within.

## Current State
While attempting to make the simulation framework more robust to unknown datasets and making the RL environment files
more flexible for building additional environments, our RL agents were unable to continue improving their performance.
I have carried on cleaning up the existing repository (this is a copy of our original repository now) while attempting
to find the broken piece in the overall framework while pushing myself to continue learning about reinforcement learning.

## Next Steps/Future Work
- [ ] Revert the simulation framework to the previous version while maintaining its currently flexibility.
- [ ] Make project easier to run locally
- [ ] Clean up the inheritance hierarchy. Some of this should be inherited while other classes should be passed as
arguments.
- [ ] Create a more well-defined optimal execution reward function. We were never able to fully agree on the structure
of the optimal execution environment which hindered our ability to evaluate our performance.

## Potential Improvements

- [ ] Abstract out the simulation framework
  - While it makes sense to keep this in the current project. I have currently been working in simulating underliers
  using Scala and learning about the Akka framework. Making this a separate project would be a great way to implement
  this knowledge while improving the training speed of the RL agents - a major bottleneck, currently, is our simulation
  framework.

## Running Locally
1. `git clone https://github.com/kew96/MicroPriceTradingRL.git`
2. Install requirements
   1. Pip: `pip install -r requirements.txt`
   2. Pipenv: `pipenv install`
   3. Conda: `conda env create -f environment.yml`
3. Install `micro_price_trading` if you wish to run the notebooks without having to add to your PATH with
`pip install -e .`. While the `-e` flag is not necessary, I recommend using it if you plan to modify any files within
`micro_price_trading/`