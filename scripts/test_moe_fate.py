import json
import pathlib
import tempfile
import unittest
import moe_fate as f

class FateTests(unittest.TestCase):
    def test_arc_replay_and_corruption(self):
        state = dict(event='initial', layer=0, use_counter=2, decay=64, slots=[0, 1], states=[2, 2],
                     hotness=[1, 1, 0, 0], last_use=[1, 2], seen=[1, 1, 0, 0],
                     arc_t1=[1, 0], arc_t2=[], arc_b1=[], arc_b2=[], arc_pending=[], arc_p=0)
        rows = [state]
        for call, expert, replacement in ((1, 0, None), (2, 2, (1, 1)), (3, 1, (0, 0)), (4, 0, (1, 2))):
            rows.append(dict(layer=0, call=call, phase='decode', ids=[expert], weights=[1]))
            if replacement:
                slot, old = replacement
                rows.append(dict(event='replace', layer=0, call=call, slot=slot, old=old, expert=expert))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'trace.jsonl'
            path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
            self.assertEqual(f.replay(path, 'fate-arc'), dict(hit=1, miss=3))
            rows[-1]['old'] = 3
            path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
            with self.assertRaises(ValueError):
                f.replay(path, 'fate-arc')

    def test_preload_first_access(self):
        state = dict(arc_t1=[0], arc_t2=[], arc_b1=[], arc_b2=[], arc_pending=[0], arc_p=0)
        arc = f.Arc(state)
        self.assertEqual(arc.access(0, [0, -1], {0}), 0)
        self.assertEqual(arc.q['t1'], [0])
        self.assertEqual(arc.q['pending'], [])
        arc.access(0, [0, -1], {0})
        self.assertEqual(arc.q['t2'], [0])

    def test_normalize(self):
        self.assertEqual(f.normalize(['exe', '-n', '256', '--moe-stream-policy', 'hotness', '-f', 'prompt']), ['-n', '256'])

if __name__ == '__main__':
    unittest.main()
