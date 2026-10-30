"""Verify that inference waiting consumes wall time but no physics steps."""
import json
from pathlib import Path
import tempfile
import unittest
from .clock_control import materialize_client


class ClockControlTests(unittest.TestCase):
    def test_waiting_does_not_advance_the_environment(self):
        source = '''def main():
    for ep in range(1):
            timestamp = 0
            actions_buffer = []
            # Episode loop.
            while True:
                if time.polls >= 3:
                    actions_buffer.append(1)
                if actions_buffer:
                    action = actions_buffer.pop()
                    pressed_digits = env.step(action)
                    in_stay_state = False
                else:
                    pressed_digits = env.stay(continue_stay=in_stay_state)
                    in_stay_state = True
                timestamp += 1
                if env.is_done:
                    break
'''
        class Clock:
            polls = 0
            @staticmethod
            def monotonic():return Clock.polls * 0.001
            @staticmethod
            def sleep(seconds):Clock.polls += 1
        class Environment:
            is_done = False
            is_success = True
            steps = 0
            def step(self, action):self.steps += 1;self.is_done = True
            def stay(self, **kwargs):raise AssertionError('Waiting advanced physics')
        class Process:
            @staticmethod
            def is_alive():return True
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/'source.py').write_text(source)
            materialize_client(path/'source.py', path/'client.py')
            env = Environment()
            namespace = dict(__name__='test_client',time=Clock,env=env,inference_proc=Process(),output_dir=path)
            exec(compile((path/'client.py').read_text(),str(path/'client.py'),'exec'),namespace)
            namespace['main']()
            result = json.loads((path/'clock_audit.json').read_text())
            self.assertEqual(result['physics_steps'],1)
            self.assertEqual(env.steps,1)
            self.assertEqual(Clock.polls,3)
            self.assertGreater(result['inference_wait_seconds'],0)

    def test_unknown_client_layout_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path/'source.py').write_text('def main():pass\n')
            with self.assertRaises(ValueError):materialize_client(path/'source.py',path/'client.py')


if __name__ == '__main__':unittest.main()
