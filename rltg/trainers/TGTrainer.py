import logging

from gym import Env

from rltg.agents.TGAgent import TGAgent
from rltg.trainers.GenericTrainer import GenericTrainer
from rltg.utils.StatsManager import StatsManager
from rltg.utils.StoppingCondition import GoalPercentage, CheckAutomataInFinalState


class TGTrainer(GenericTrainer):
    """Temporal Goal Trainer"""

    def __init__(self, env:Env, agent:TGAgent=None, n_episodes=1000,
                 resume=False,
                 eval=False,
                 data_dir="data",
                 stop_conditions=(GoalPercentage(10, 1.0), ),
                 renderer=None
                 ):
        super().__init__(env, agent, n_episodes, resume, eval, data_dir, stop_conditions, renderer=renderer)
        self.stop_conditions.append(CheckAutomataInFinalState())

    def train_loop(self):
        env = self.env
        agent = self.agent

        stop_condition = False
        info = {"goal": False}

        state = env.reset()
        action = agent.start(state)
        obs = None
        if self.renderer: self.renderer.update(env)

        while not stop_condition:
            state2, reward, done, info = env.step(action)
            if self.renderer: self.renderer.update(env)

            old_automata_state = agent.get_automata_state()
            new_automata_state = agent.sync(action, state2)

            stop_condition = self.check_episode_stop_conditions(done, info.get("goal", False))
            obs = agent.observe(state, action, reward, state2,
                                old_automata_state=old_automata_state,
                                new_automata_state=new_automata_state,
                                is_terminal_state=stop_condition)

            if stop_condition:
                break

            action = agent.step(obs)
            agent.update()
            state = state2

        agent.end(obs)
        steps, tot_reward = agent.brain.episode_iteration, agent.brain.total_reward
        return steps, tot_reward, self.is_goal(info)


    def check_stop_conditions(self, stats:StatsManager, *args, **kwargs):
        return not self.eval and \
               all([s.check_condition(stats_manager=stats, temporal_evaluators=self.agent.temporal_evaluators)
                    for s in self.stop_conditions])

    def check_episode_stop_conditions(self, done, goal, *args, **kwargs):
        temp_evals = self.agent.temporal_evaluators
        any_te_failed = any(t.is_failed() for t in temp_evals)
        all_te_true = all(t.is_true() for t in temp_evals) if len(temp_evals) > 0 else False

        if any_te_failed:
            logging.debug("some automaton in failure state: end episode")

        return done or any_te_failed or (goal and all_te_true)

    def is_goal(self, info, *args, **kwargs):
        return super().is_goal(info) and all(t.is_true() for t in self.agent.temporal_evaluators)

    def load_agent(self):
        return TGAgent.load(self.agent_data_dir)

