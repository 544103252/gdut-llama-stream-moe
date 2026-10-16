"""Deterministic EAMC and prediction/replacement replay tests."""
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import moe_infinity as m
import numpy as np


class InfinityTests(unittest.TestCase):
    def test_calibration_request_boundaries_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            model = root / "model.gguf"
            model.write_bytes(b"abc")
            (root / "calibration").mkdir()
            samples = [dict(id="train-a", split="train", prompt="a"),
                       dict(id="train-b", split="train", prompt="b"),
                       dict(id="test-c", split="test", prompt="c")]
            (root / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
            for i, sample in enumerate(samples[:2]):
                stem = root / "calibration" / sample["id"]
                stem.with_suffix(".status.json").write_text(json.dumps(dict(
                    returncode=0, sample=sample["id"], command=["exe", "-m", str(model)])), encoding="utf-8")
                stem.with_suffix(".prompt.txt").write_text(sample["prompt"], encoding="utf-8")
                rows = [dict(phase="prefill", layer=0, step=0, ids=[0]),
                        dict(phase="decode", layer=0, step=1, ids=[i, i])]
                stem.with_suffix(".jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with patch.object(m.m, "model_dims", return_value=(1, 2)):
                m.calibrate(root, model)
                history, _ = m.load(root / "eam.bin")
                np.testing.assert_array_equal(history, [[[2, 0]], [[0, 2]]])
                status_path = root / "calibration" / "train-a.status.json"
                status = json.loads(status_path.read_text())
                status["sample"] = "test-c"
                status_path.write_text(json.dumps(status), encoding="utf-8")
                with self.assertRaises(ValueError):
                    m.calibrate(root, model)

    def test_matching_partial_layers_and_capacity(self):
        a = np.array([[4, 0], [0, 3]], dtype=np.uint32)
        b = np.array([[0, 4], [3, 0]], dtype=np.uint32)
        query = np.array([[1, 0], [0, 0]], dtype=np.uint32)
        self.assertEqual(m.select(np.array([a, b]), query), 0)
        self.assertEqual(m.select(np.array([a, a]), query), 0)
        self.assertEqual(m.select(np.array([a, b]), query * 0), -1)
        history = []
        m.insert(history, a, 2)
        m.insert(history, b, 2)
        m.insert(history, a * 2, 2)
        np.testing.assert_array_equal(history[0], a * 2)
        np.testing.assert_array_equal(history[1], b)
        with self.assertRaises(ValueError):
            m.insert(history, a * 0, 2)
        with self.assertRaises(ValueError):
            m.insert(history, a, 0)

    def test_format_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            eam = root / "eam.bin"
            payload = m.HEADER.pack(b"MOEEAM01", 1, 4, 1, b"a" * 32)
            payload += np.array([4, 2, 1, 1], dtype="<u4").tobytes()
            eam.write_bytes(payload)
            history, digest = m.load(eam)
            self.assertEqual(history.shape, (1, 1, 4))
            self.assertEqual(digest, (b"a" * 32).hex())
            for broken in (payload[:10], payload[:-1], payload + b"x", b"BADVERSN" + payload[8:],
                           payload[:m.HEADER.size] + bytes(16)):
                eam.write_bytes(broken)
                with self.assertRaises(ValueError):
                    m.load(eam)
            eam.write_bytes(payload)
            rows = [dict(event="eam", layer=0, call=1, match=0, fallback="none"),
                    dict(event="initial", layer=0, slots=[3, 1, 2], states=[2, 2, 2],
                         hotness=[1, 10, 5, 20], last_use=[0, 1, 2], use_counter=2, decay=0),
                    dict(layer=0, call=1, phase="decode", ids=[0], weights=[1]),
                    dict(event="replace", layer=0, call=1, slot=2, old=2, expert=0),
                    dict(event="eam", layer=0, call=2, match=0, fallback="none"),
                    dict(layer=0, call=2, phase="decode", ids=[0], weights=[1])]
            trace = root / "trace.jsonl"
            trace.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            self.assertEqual(m.replay(trace, eam), dict(hit=1, miss=1))
            rows[3]["slot"] = 0
            trace.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                m.replay(trace, eam)


if __name__ == "__main__":
    unittest.main()
