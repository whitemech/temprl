# History

## 0.4.0 (2021-05-19)

* The package now requires Python 3.8+. This allowed to upgrade 
  some dependencies like `gym>=0.17.2` and `numpy>=1.22.2`.
* Used `pythomata` at version `>=0.3.0`.
* Introduced better abstract classes and interfaces for supporting reward machines.
  * The base classes and interfaces can be found in `temprl.reward_machines.base`.
    `AbstractRewardMachine` is not coupled with `pythomata.DFA`, giving the developer
    more flexibility on how to implement its own reward machine. Analogously,
    `AbstractRewardMachineSimulator` is a pure interface for the simulation of a 
    reward machine, which is ultimately used by the `TemporalGoal` class.
    Anyway, the library provides a concrete class `RewardMachineSimulator(AbstractRewardMachineSimulator)`
    which should satisfy most of the use cases regarding the RM simulator.
  * In `temprl.reward_machines.automata`, the library provides an implementation of `AbstractRewardMachine`,
    called `RewardAutomaton`, which is based on `pythomata.DFA` to represent the reward machine structure.
* Introduced `AbstractStepController`, which allows to filter out certain fluents observations from the RM perspective.
  The library provides two implementations: `StatelessStepController`, which behaves the same regardless of the 
  history of the trajectory, and `StatefulStepController`, which is based on a `pythomata.DFA` instance and accepts
  a fluents configuration if after reading it the simulation is in an accepting state.
* Improved linting by adding Pylint checks.
* Other misc fixes to minor issues.


## 0.3.0 (2021-06-20)

* Simplify APIs of `TemporalWrapper`: remove `feature_extractor` and 
  `combine` parameters, as well as reward shaping support.
  The reason is that these functionalities, in the OpenAI Gym "philosophy",
  should be delegated to other Gym wrappers, e.g. `ObservationWrapper`
  for combining the features and the automata states.
* Remove `flloat` dependency. Since `TemporalGoal` now only requires 
  a `pythomata.DFA` object, it is up to the user to decide how to 
  generate the reward automaton.
* Update dependencies to their latest version, e.g. `pythomata`.
* The `reset()` method of the temporal wrapper now first resets
  the temporal goals, and then makes a step on each of them
  according to the fluents extracted from the environment's initial
  state. This is needed because otherwise the initial state
  of the wrapped environment is ignored.
* The support for terminating conditions from the temporal goals
  is removed. Again, this is because the only job of the 
  DFAs is to provide rewards according to the history of the
  episode; any other customization of the underlying envrionment,
  or the wrapper, must be done by using other wrappers. 

## 0.2.3 (2020-09-24)

* Let temporal wrapper initialization directly from DFA
* Added 'combine' method to customize how to do the actual
  product between DFA states and the environment state space.
* Added new linters.
* Renewed documentation.

## 0.2.2 (2019-11-03)

* Improved support for reward shaping.
* Fixed minor bugs and issues.

## 0.2.1 (2019-11-03)

* Improved testing and continuous integration.
* Fixed minor bugs and issues.

## 0.2.0 (2019-07-13)

* Main refactoring of the package.

## 0.1.0 (2018-04-04)

* First release on PyPI.
