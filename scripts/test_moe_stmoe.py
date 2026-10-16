"""Deterministic tests for calibration, replay, metrics and experiment isolation."""
import copy
import json
import pathlib
import tempfile
import unittest

import moe_stmoe as m


class StmoeTest(unittest.TestCase):
    def test_predict_and_update(self):
        rows = [[[0, 1], [1, 0]], [[0, 1], [2, 3]], [], [], []]
        self.assertEqual(m.predict(rows, [0, 1], []), [0, 2])
        self.assertEqual(m.predict(rows, [0, 1], [1, 2]), [0, 1, 2])
        self.assertEqual(m.predict(rows, [2], []), [])
        m.update(rows, [0, 1], [3, 2])
        self.assertEqual(rows[0], [[0, 0], [2, 2]])
        self.assertEqual(rows[1], [[0, 0], [2, 3]])
        m.update(rows, [0], [3, 2])
        self.assertEqual(rows[0], [[3, 2], [2, 3]])
        full = [[[0, 0], [1, 0]]]
        m.update(full, [0], [1])
        self.assertEqual(full, [[[0, 0], [1, 1]]])

    def test_calibration_and_trace_validation(self):
        meta = dict(layers=[0, 2], experts=4, k=2)
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "trace.jsonl"
            rows = [dict(layer=il, step=step, phase="decode", ids=ids)
                    for step in range(2) for il, ids in ((0, [0, 1]), (2, [3, 2]))]
            prefill = [dict(layer=0, step=0, phase="prefill", ids=[2, 3])]
            path.write_text("\n".join(map(json.dumps, prefill + rows)), encoding="utf-8")
            tables, tokens = m.build_cct(meta, [path])
            self.assertEqual(tokens, 2)
            self.assertEqual(tables[0]["rows"], [[[2, 2], [3, 2]], [[2, 2], [3, 2]], [], []])
            for invalid in (rows[:-1], rows[1:], [dict(rows[0], ids=[0, 0]), *rows[1:]],
                            [*rows[:2], dict(rows[2], step=5), rows[3]], []):
                path.write_text("\n".join(map(json.dumps, invalid)), encoding="utf-8")
                with self.assertRaises(ValueError):
                    m.build_cct(meta, [path])

    def test_replay_history_and_tampering(self):
        cct = dict(layers=[0, 2], k=2, pairs=[dict(source=0, target=2, experts=4,
                   rows=[[[0, 2], [1, 2]], [], [], []])])
        original = copy.deepcopy(cct)
        events = []
        for step, predicted in enumerate(([0, 1], [2, 3])):
            events += [dict(event="stmoe_predict", source=0, target=2, step=step, predicted=predicted),
                       dict(layer=0, step=step, phase="decode", ids=[0, 1]),
                       dict(event="stmoe_evaluate", source=0, target=2, step=step, predicted=predicted,
                            actual=[2, 3], intersection=[] if step == 0 else [2, 3]),
                       dict(layer=2, step=step, phase="decode", ids=[2, 3])]
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "trace.jsonl"
            path.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
            result = m.replay(path, cct)
            self.assertEqual(m.metrics(result[2]), dict(predicted=4, actual=4, hit=2, miss=2,
                             false_positive=2, all_hit=1, observations=2, coverage=0.5, precision=0.5,
                             all_hit_rate=0.5, mean_predicted=2.0))
            self.assertEqual(cct, original)
            events[0]["predicted"] = [2, 3]
            path.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
            with self.assertRaises(ValueError):
                m.replay(path, cct)

    def test_manifest_and_disjointness(self):
        calibration = [dict(id="train-0", prompt_sha256="a")]
        m.ensure_disjoint(calibration, [dict(id="test-0", prompt_sha256="b")])
        for row in (dict(id="train-0", prompt_sha256="b"), dict(id="test-0", prompt_sha256="a")):
            with self.assertRaises(ValueError):
                m.ensure_disjoint(calibration, [row])
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "manifest.json"
            m.pin(path, {"model": "a"})
            m.pin(path, {"model": "a"})
            with self.assertRaises(ValueError):
                m.pin(path, {"model": "b"})

    def test_zero_denominators(self):
        result = m.metrics({})
        self.assertIsNone(result["coverage"])
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["all_hit_rate"])
        self.assertIsNone(result["mean_predicted"])
        result = m.metrics(dict(actual=2, observations=1))
        self.assertEqual(result["coverage"], 0)
        self.assertIsNone(result["precision"])


if __name__ == "__main__":
    unittest.main()
