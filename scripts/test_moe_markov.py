"""Offline tests for calibration and live replacement replay."""
import json
import pathlib
import struct
import tempfile
import unittest

import moe_markov as m
import numpy as np


class ExperimentTests(unittest.TestCase):
    def test_rank_buckets_repeats_loading_and_short_cache(self):
        counts = dict(counts=[0]*180, miss=0, loading=0, overflow=0)
        slots, states = [2, 0, 1, 3], [2, 2, 2, 1]
        hotness, last = [5, 5, 9, 100], [1, 10, 20, 99]
        m.count_ranks(slots, states, hotness, last, [0, 2, 4, 0, 1, 3], counts)
        self.assertEqual(counts["counts"][:3], [1, 1, 2])
        self.assertEqual(sum(counts["counts"][3:]), 0)
        self.assertEqual((counts["miss"], counts["loading"]), (2, 1))
        self.assertEqual(hotness, [5, 5, 9, 100])
        slots[0] = 4
        hotness.append(12)
        m.count_ranks(slots, states, hotness, last, [4], counts)
        self.assertEqual(counts["counts"][0], 2)
        last[2] = last[1]
        m.count_ranks(slots, states, hotness, last, [0], counts)
        self.assertEqual(counts["counts"][1], 2)
        overflow = dict(counts=[0]*180, miss=0, loading=0, overflow=0)
        m.count_ranks(list(range(181)), [2]*181, list(range(181,0,-1)), [0]*181,
                      [179, 180], overflow)
        self.assertEqual(overflow["counts"][179], 1)
        self.assertEqual(overflow["overflow"], 1)

    def test_weighted_calibration_and_sample_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "calibration").mkdir()
            (root / "model").write_bytes(b"abc")
            (root / "sources.json").write_text("[]")
            rows = [dict(layer=0, phase="decode", ids=[0, 1], weights=[.75, .25]),
                    dict(layer=0, phase="decode", ids=[2, 3], weights=[.5, .5])]
            for i in range(200):
                (root / "calibration" / f"{i}.jsonl").write_text("\n".join(map(json.dumps, rows)))
            old = m.model_dims
            m.model_dims = lambda _: (1, 4)
            try:
                m.calibrate(root, root / "model")
            finally:
                m.model_dims = old
            payload = (root / "markov.bin").read_bytes()
            self.assertEqual(payload[:8], b"MOEMKV01")
            self.assertEqual(payload[16:48].hex(), m.digest(root / "model"))
            matrix = np.frombuffer(payload[48:], dtype="<f4").reshape(4, 4)
            expected = np.full((4, 4), .1)
            expected[0, 2:] += 75
            expected[1, 2:] += 25
            expected /= expected.sum(-1, keepdims=True)
            np.testing.assert_allclose(matrix, expected, rtol=1e-6)
            np.testing.assert_allclose(matrix[2:], .25)

    def test_replay_checks_replacement_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            matrix = np.full((1, 4, 4), .25, dtype="<f4")
            (root / "matrix").write_bytes(struct.pack("<8sII32s", b"MOEMKV01", 1, 4, bytes(32)) + matrix.tobytes())
            initial = dict(event="initial", layer=0, use_counter=3, decay=64,
                           slots=[3, 1, 2], states=[2, 2, 2], hotness=[0, 10, 5, 20],
                           last_use=[1, 2, 3], seen=[0, 1, 1, 1])
            row = dict(layer=0, phase="decode", call=4, ids=[0, 1], weights=[.75, .25])
            # Equal chain scores: active expert 1 is protected, so expert 2 wins.
            replacement = dict(event="replace", layer=0, call=4, slot=2, old=2, expert=0)
            trace = root / "trace"
            trace.write_text("\n".join(map(json.dumps, [initial, row, replacement])))
            for policy in ("chain", "hotness"):
                ranks = {}
                self.assertEqual(m.replay_decode(trace, root / "matrix", policy, ranks), dict(hit=1, miss=1))
                self.assertEqual(ranks[0]["counts"][1], 1)
                self.assertEqual(ranks[0]["miss"], 1)
            replacement["slot"] = 0
            trace.write_text("\n".join(map(json.dumps, [initial, row, replacement])))
            with self.assertRaisesRegex(RuntimeError, "replacement mismatch"):
                m.replay_decode(trace, root / "matrix", "chain")


if __name__ == "__main__":
    unittest.main()
