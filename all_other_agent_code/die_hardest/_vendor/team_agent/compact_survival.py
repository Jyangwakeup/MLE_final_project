"""Decision-local exact transition background and integer-grid viability."""
from itertools import product
import numpy as np
import settings as s
from .danger import blast_coords
from .opponent_transitions import physical_actions, _execution_orders, _apply_action_phase, _progress_world
from .temporal_safety_features import temporal_maps


class TransitionContext:
    def __init__(self, state, stats):
        self.state = state
        self.stats = stats
        self.actors = (tuple(state['self']), *map(tuple, state['others']))
        self.field = np.asarray(state['field'])
        self.bombs = tuple((tuple(p), int(t)) for p, t in state['bombs'])
        occupied = frozenset(a[3] for a in self.actors)
        bomb_positions = frozenset(p for p, _ in self.bombs)
        self.choices = tuple(physical_actions(self.field, tuple(a[3]), bool(a[2]), occupied, bomb_positions) for a in self.actors[1:])
        self.background, _ = _progress_world(state, self.actors, self.bombs)
        self.danger = np.asarray(state['explosion_map']) > 0
        for p, timer in self.bombs:
            if timer <= 0:
                for q in blast_coords(self.field, p, s.BOMB_POWER):
                    self.danger[q] = True
        stats['world_background_builds'] = stats.get('world_background_builds', 0) + 1

    def scenarios(self, action, check):
        seen = set()
        for opponents in product(*self.choices):
            check()
            profile = (action, *opponents)
            for order in _execution_orders(self.actors, profile):
                check()
                self.stats['action_phase_simulations'] = self.stats.get('action_phase_simulations', 0) + 1
                actors, bombs = _apply_action_phase(self.state, profile, order)
                alive = not bool(self.danger[actors[0][3]])
                others = tuple(a for a in actors[1:] if not self.danger[a[3]])
                remaining = tuple((p, t - 1) for p, t in bombs if t > 0)
                key = (alive, actors[0][3], bool(actors[0][2]),
                       tuple(sorted((str(a[0]), tuple(a[3]), bool(a[2])) for a in others)),
                       tuple(sorted(remaining)))
                if key in seen:
                    continue
                seen.add(key)
                self.stats['distinct_scenarios'] = self.stats.get('distinct_scenarios', 0) + 1
                yield actors[0], others, remaining, alive, profile, order


class BitGrid:
    def __init__(self, field):
        self.field = np.asarray(field)
        self.width, self.height = self.field.shape
        self.full = (1 << self.field.size) - 1
        self.low = sum(1 << (x * self.height) for x in range(self.width))
        self.high = self.low << (self.height - 1)
        self.blasts = tuple(self.pack_positions(blast_coords(self.field, (x, y), s.BOMB_POWER))
                            for x in range(self.width) for y in range(self.height))

    def bit(self, position):
        return 1 << (int(position[0]) * self.height + int(position[1]))

    def pack_positions(self, positions):
        result = 0
        for position in positions:
            result |= self.bit(position)
        return result

    def pack(self, array):
        return int.from_bytes(np.packbits(np.asarray(array, dtype=bool).ravel(), bitorder='little').tobytes(), 'little')

    def expand(self, bits):
        return (bits | (bits << self.height) | (bits >> self.height)
                | ((bits & ~self.high) << 1) | ((bits & ~self.low) >> 1)) & self.full

    def blast_union(self, positions):
        result = 0
        while positions:
            bit = positions & -positions
            result |= self.blasts[bit.bit_length() - 1]
            positions ^= bit
        return result


class ViabilityContext:
    def __init__(self, transitions, steps, rearming, check, stats):
        self.transitions = transitions
        self.steps = steps
        self.rearming = rearming
        self.check = check
        self.stats = stats
        self.grid = BitGrid(transitions.background['field'])
        self.maps = {}
        self.memo = {}

    def winning(self, own, others, bombs):
        self.check()
        if self.steps <= 0:
            return True, 0
        # Match legacy canonical identity, including named opponents/capacity.
        bomb_key = tuple(sorted(bombs))
        key = (bomb_key, tuple(sorted((str(a[0]), tuple(a[3]), bool(a[2])) for a in others)))
        cached = self.memo.get(key)
        if cached is not None:
            self.stats['viability_cache_hits'] = self.stats.get('viability_cache_hits', 0) + 1
            return bool(cached[0] & self.grid.bit(own[3])), cached[1]
        maps = self.maps.get(bomb_key)
        if maps is None:
            state = {**self.transitions.background, 'self': own, 'others': (), 'bombs': bombs}
            danger, blocked = temporal_maps(state, max(1, self.steps), hypothetical_bomb=False)
            maps = ([self.grid.pack(a) for a in danger], [self.grid.pack(a) for a in blocked])
            self.maps[bomb_key] = maps
            self.stats['temporal_map_builds'] = self.stats.get('temporal_map_builds', 0) + 1
        else:
            self.stats['temporal_map_cache_hits'] = self.stats.get('temporal_map_cache_hits', 0) + 1
        danger, initial_blocked = maps
        blocked = list(initial_blocked)
        # Legacy maps mark an opponent standing on a bomb blocked at ALL times;
        # its special opponent-clearing branch does not clear that position.
        bomb_positions = {p for p, _ in bombs}
        permanent = self.grid.pack_positions(a[3] for a in others if tuple(a[3]) in bomb_positions)
        if permanent:
            blocked = [b | permanent for b in blocked]
        named = {str(a[0]): tuple(a[3]) for a in others}
        armed_names = {str(a[0]) for a in others if self.rearming or bool(a[2])}
        reach = [self.grid.pack_positions(a[3] for a in others)]
        armed = [self.grid.pack_positions(a[3] for a in others if self.rearming or bool(a[2]))]
        frontier = self.grid.pack_positions(named.values())
        armed_frontier = self.grid.pack_positions(p for name, p in named.items() if name in armed_names)
        for t in range(1, self.steps + 1):
            frontier = self.grid.expand(frontier) & ~blocked[t]
            armed_frontier = self.grid.expand(armed_frontier) & ~blocked[t]
            reach.append(frontier)
            armed.append(armed_frontier)
        robust = list(danger)
        for t in range(1, self.steps + 1):
            explosion = t + int(s.BOMB_TIMER)
            if explosion > self.steps:
                break
            blast = self.grid.blast_union(armed[t - 1])
            robust[explosion] |= blast
            if explosion + 1 <= self.steps:
                robust[explosion + 1] |= blast
        viable = self.grid.full & ~(blocked[self.steps] | reach[self.steps] | robust[self.steps])
        examined = viable.bit_count()
        for t in range(self.steps - 1, -1, -1):
            allowed = self.grid.full & ~(blocked[t] | (reach[t] if t else 0) | robust[t])
            enterable = viable & ~blocked[t + 1]
            viable = allowed & (viable | self.grid.expand(enterable))
            examined += allowed.bit_count()
            self.check()
        self.memo[key] = viable, examined
        self.stats['viability_board_builds'] = self.stats.get('viability_board_builds', 0) + 1
        return bool(viable & self.grid.bit(own[3])), examined
