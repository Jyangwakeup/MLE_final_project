"""Disjoint, event-labelled Task4 replay. No reward or safety reinterpretation."""
from collections import deque
from agent_code.expected_sarsa._vendor.dqn_model import ReplayBuffer, REPLAY_CHECKPOINT_FORMAT

VERSION = 'task4-kill-replay-v1'
CONTRACT = dict(version=VERSION, ordinary_capacity=16000, kill_capacity=4000,
                preferred_kill_samples=16, batch_size=64, label='nstep_observed_kill_or')

class KillReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, seed):
        if capacity != 20000: raise ValueError('Kill replay capacity must be 20000')
        super().__init__(capacity, seed)
        self.ordinary = deque(maxlen=16000)
        self.kills = deque(maxlen=4000)
        self.counters = dict(ordinary_evicted=0, kill_evicted=0, batches=0, kill_samples=0, ordinary_samples=0)

    def configure(self, current_task, per_task_capacity=None):
        if current_task != 'full_match' or per_task_capacity not in (None, 20000):
            raise ValueError('Kill replay only accepts full_match / 20000')
        super().configure(current_task, per_task_capacity)

    def append(self, transition):
        if transition.task_id != 'full_match': raise ValueError('Foreign kill replay task')
        pool = self.kills if transition.observed_kill else self.ordinary
        if len(pool) == pool.maxlen:
            self.counters['kill_evicted' if transition.observed_kill else 'ordinary_evicted'] += 1
        pool.append(transition)

    def __len__(self): return len(self.ordinary) + len(self.kills)
    def current_size(self): return len(self)
    def _all_items(self): return list(self.ordinary) + list(self.kills)

    def sample_batch(self, size, parent_fraction=0., safety_replay_spec=None,
                     task_samples=None, sampling_version=None):
        if (size != 64 or parent_fraction != 0 or (safety_replay_spec or {}).get('enabled')
                or task_samples is not None or sampling_version != VERSION or len(self) < size):
            raise ValueError('Kill replay sampling contract mismatch')
        count = min(16, len(self.kills))
        ordinary_count = min(size-count, len(self.ordinary))
        count = size-ordinary_count
        items = self.random.sample(list(self.kills), count) + self.random.sample(list(self.ordinary), ordinary_count)
        self.random.shuffle(items)
        self.counters['batches'] += 1
        self.counters['kill_samples'] += count
        self.counters['ordinary_samples'] += ordinary_count
        return self.pack_batch(items, size)

    def state_dict(self):
        value = super().state_dict()
        value.update(format=VERSION, kill_contract=CONTRACT,
                     observed_kills=[bool(t.observed_kill) for t in self._all_items()],
                     kill_counters=dict(self.counters), occupancy=dict(ordinary=len(self.ordinary), kills=len(self.kills)))
        return value

    def load_state_dict(self, state):
        if (state.get('format') != VERSION or state.get('kill_contract') != CONTRACT
                or state.get('capacity') != 20000 or state.get('current_task') != 'full_match'
                or len(state.get('observed_kills', [])) != state.get('count')
                or set(state.get('task_ids', [])) - {'full_match'}):
            raise ValueError('Missing or incompatible kill replay state')
        labels = state['observed_kills']; ordinary = labels.count(False); kills = labels.count(True)
        if (any(type(x) is not bool for x in labels) or labels != [False]*ordinary+[True]*kills
                or ordinary>16000 or kills>4000 or state.get('occupancy') != dict(ordinary=ordinary,kills=kills)):
            raise ValueError('Invalid kill replay FIFO labels or occupancy')
        if set(state.get('kill_counters', {})) != set(self.counters): raise ValueError('Missing kill replay counters')
        self.ordinary.clear(); self.kills.clear()
        super().load_state_dict(dict(state, format=REPLAY_CHECKPOINT_FORMAT))
        self.counters = dict(state['kill_counters'])

    def mark_recent_own_bomb_fatal(self, task_id, count=4):
        # Safety replay is disabled in this protocol; do not relabel combat data.
        return None
