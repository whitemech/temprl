from copy import copy


class mydefaultdict(dict):
    def __init__(self, x):
        super().__init__()
        self._default = x

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        else:
            self[key] = copy(self._default)
            return self[key]


class AgentObservation(object):
    def __init__(self, state, action, reward, state2):
        self.state = state
        self.action = action
        self.reward = reward
        self.state2 = state2

    def unpack(self):
        return self.state, self.action, self.reward, self.state2


def compute_levels(dfa, property_states):
    assert property_states.issubset(dfa.states)
    level = 0
    state2level = {final_state: level for final_state in property_states}

    z_current = set()
    z_next = set(property_states)
    while z_current != z_next:
        level += 1
        z_current = z_next
        z_next = copy(z_current)
        for s in dfa.states:
            if s in z_current:
                continue
            for a in dfa.transition_function.get(s, []):
                next_state = dfa.transition_function[s][a]
                if next_state in z_current:
                    z_next.add(s)
                    state2level[s] = level

    z_current = z_next

    max_level = level - 1

    # levels for failure state (i.e. that cannot reach a final state)
    failure_states = set()
    for s in filter(lambda x: x not in z_current, dfa.states):
        state2level[s] = level
        failure_states.add(s)

    return state2level, max_level, failure_states


def _potential_function(q, initial_state, reachability_levels, reward, is_terminal_state=False):
    if is_terminal_state:
        return 0
    else:
        # p = 1/(self.reachability_levels[q]) * self.reward
        # if q == initial_state and reachability_levels[initial_state]==0:
        #     return reward
        initial_state_level = reachability_levels[initial_state]
        p = initial_state_level - reachability_levels[q]
        p = p/initial_state_level if initial_state_level!=0 else p
        p *= reward
    return p
