"""
Randomized checks of Raft's headline safety properties (paper section
5.4.3, Figure 3), run across many seeds against `SimulatedCluster`
rather than asserted for one hand-picked scenario. None of these use
real time or threads -- every seed drives a fully deterministic
simulated run, so a failure here is always exactly reproducible by
re-running with the same seed.

  Election Safety        at most one leader can be elected in a
                          given term.
  Leader Append-Only      a leader never overwrites or deletes entries
                          in its own log (only appends).
  Log Matching            if two logs contain an entry with the same
                          index and term, the logs are identical in all
                          entries up through that index.
  Leader Completeness     if a log entry is committed in a given term,
                          it will be present in the logs of the leaders
                          for all higher-numbered terms.
  State Machine Safety    if a server has applied a log entry at a
                          given index to its state machine, no other
                          server will ever apply a different log entry
                          for the same index.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from raftkv.cluster import SimulatedCluster
from raftkv.kv.store import Command, SetCommand

SEEDS = range(40)


class TestElectionSafety(unittest.TestCase):
    def test_at_most_one_leader_per_term_across_many_random_runs_with_faults(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                c = SimulatedCluster([1, 2, 3, 4, 5], seed=seed)
                rng = random.Random(f"fault:{seed}")
                leader_by_term: dict[int, int] = {}

                for tick in range(400):
                    c.tick()
                    for node_id in c.node_ids:
                        node = c.nodes[node_id]
                        if node.role.value != "leader":
                            continue
                        term = node.current_term
                        if term in leader_by_term:
                            self.assertEqual(
                                leader_by_term[term],
                                node_id,
                                f"seed={seed}: two different leaders ({leader_by_term[term]} "
                                f"and {node_id}) both claimed term {term}",
                            )
                        else:
                            leader_by_term[term] = node_id

                    # occasionally inject a fault to actually exercise
                    # re-elections instead of just watching one stable leader
                    if tick % 37 == 0 and len(c.alive) > 3:
                        victim = rng.choice(list(c.alive))
                        c.kill(victim)
                    if tick % 53 == 0:
                        for n in c.node_ids:
                            c.revive(n)


class TestLeaderAppendOnly(unittest.TestCase):
    def test_a_leaders_own_log_never_shrinks_or_changes_a_committed_entry(self):
        for seed in range(15):
            with self.subTest(seed=seed):
                c = SimulatedCluster([1, 2, 3], seed=seed)
                c.run_until(lambda cl: cl.leader() is not None, max_ticks=200)
                # node_id -> the (term, index) pairs that were committed the
                # last time we looked -- only ever compared as a *prefix* of
                # a later, possibly-longer committed snapshot, since a
                # node's own commit_index only ever advances.
                committed_snapshot: dict[int, tuple] = {n: () for n in c.node_ids}

                for i in range(1, 6):
                    leader = c.leader()
                    if leader is not None:
                        c.propose(Command("c", i, SetCommand("k", str(i))), via=leader)
                    c.run(20)
                    for node_id in c.node_ids:
                        node = c.nodes[node_id]
                        committed_entries = node.log.entries_between(1, node.commit_index)
                        current_committed = tuple((e.term, e.index) for e in committed_entries)
                        prior = committed_snapshot[node_id]
                        self.assertEqual(
                            current_committed[: len(prior)],
                            prior,
                            f"seed={seed}: node {node_id}'s previously-committed entries changed",
                        )
                        committed_snapshot[node_id] = current_committed


class TestLogMatchingProperty(unittest.TestCase):
    def test_identical_index_and_term_implies_identical_entries_up_to_that_point(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                c = SimulatedCluster([1, 2, 3], seed=seed)
                c.run_until(lambda cl: cl.leader() is not None, max_ticks=200)
                for i in range(1, 8):
                    leader = c.leader()
                    if leader is not None:
                        c.propose(Command("c", i, SetCommand("k", str(i))), via=leader)
                    c.run(15)
                    if i == 4 and len(c.alive) > 2 and c.leader() is not None:
                        # force a mid-stream leadership change
                        c.kill(c.leader())
                        c.run(40)

                # Compare every pair of nodes: wherever their logs agree on
                # (term, index) for some entry, every earlier entry must
                # match too.
                for a in c.node_ids:
                    for b in c.node_ids:
                        if a >= b:
                            continue
                        log_a, log_b = c.nodes[a].log, c.nodes[b].log
                        upto = min(log_a.last_index, log_b.last_index)
                        agreement_point = None
                        for idx in range(upto, 0, -1):
                            if log_a.term_at(idx) == log_b.term_at(idx) and log_a.term_at(idx) != 0:
                                agreement_point = idx
                                break
                        if agreement_point is None:
                            continue
                        for idx in range(1, agreement_point + 1):
                            self.assertEqual(
                                log_a.get(idx).command,
                                log_b.get(idx).command,
                                f"seed={seed}: logs of {a} and {b} agree on term at index "
                                f"{agreement_point} but diverge at index {idx}",
                            )


class TestStateMachineSafety(unittest.TestCase):
    def test_every_node_applies_the_same_command_at_the_same_index(self):
        for seed in range(25):
            with self.subTest(seed=seed):
                c = SimulatedCluster([1, 2, 3, 4, 5], seed=seed)
                rng = random.Random(f"chaos:{seed}")
                c.run_until(lambda cl: cl.leader() is not None, max_ticks=200)

                for i in range(1, 21):
                    leader = c.leader()
                    if leader is not None:
                        c.propose(Command("c", i, SetCommand("k", str(i))), via=leader)
                    c.run(rng.randint(3, 10))
                    if rng.random() < 0.15 and len(c.alive) > 3:
                        c.kill(rng.choice(list(c.alive)))
                    if rng.random() < 0.1:
                        dead = set(c.node_ids) - c.alive
                        if dead:
                            c.revive(rng.choice(list(dead)))

                c.run(200)  # let everything quiesce

                agreed_index = min(c.nodes[n].commit_index for n in c.node_ids)
                for idx in range(1, agreed_index + 1):
                    commands_at_idx = {
                        c.nodes[n].log.get(idx).command for n in c.node_ids
                    }
                    self.assertEqual(
                        len(commands_at_idx),
                        1,
                        f"seed={seed}: nodes disagree on the committed command at index {idx}",
                    )


if __name__ == "__main__":
    unittest.main()
