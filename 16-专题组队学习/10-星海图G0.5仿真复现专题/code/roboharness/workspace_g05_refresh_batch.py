"""Collect pending candidate results without changing the observation."""
NAME = 'g05_refresh_batch'
SCHEMA = {'type': 'object', 'properties': {'batch_id': {'type': 'string'}},
          'required': ['batch_id'], 'additionalProperties': False}

if __name__ == '__main__':
    raise RuntimeError('g05_refresh_batch requires the orchestrated host')
