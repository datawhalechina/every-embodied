"""Publish bounded development requests and import observation-bound proposals.

The controller supplies an already staged causal capture and the current
observation binding. A separately registered development worker owns inference;
this exchange never loads a model or sends a physical action. Qualification
evidence and finite budgets must be frozen before the first publication.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from bridge import joint_tool_arguments


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive(path, blob):
    """Make a complete file visible once, without replacing another writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def contained(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Artifact path must be relative')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes its registered root')
    return path


class ProposalExchange:
    """Persist one inference reservation per exact input and one episode budget.

    Registration is created by the future development supervisor after physical
    qualification. Its worker directory must differ from the qualification
    directory. Caller-owned evidence remains outside the model workspace.
    """

    def __init__(self, registration, evidence, workspace):
        self.registration = Path(registration).resolve()
        self.evidence, self.workspace = Path(evidence).resolve(), Path(workspace).resolve()
        self.blob = self.registration.read_bytes()
        self.config = config = json.loads(self.blob)
        if config.get('mode') != 'development' or config.get('native_step_cap') != 27664:
            raise ValueError('A frozen development registration is required')
        if type(config.get('model_call_limit')) is not int or not 1 <= config['model_call_limit'] <= 27664:
            raise ValueError('An explicit finite development model-call budget is required')
        if not isinstance(config.get('episode_id'), str) or not config['episode_id']:
            raise ValueError('An exact development episode is required')
        self.worker = Path(config['worker_root']).resolve()
        qualification_root = Path(config['qualification_worker_root']).resolve()
        if self.worker == qualification_root or self.worker.is_relative_to(qualification_root):
            raise ValueError('Development requests cannot consume the qualification worker budget')
        self.input_root = self.worker / 'inputs'
        if self.evidence.is_relative_to(self.workspace) or self.worker.is_relative_to(self.workspace):
            raise ValueError('Worker and exchange evidence must be outside the model workspace')
        if self.evidence == self.worker or self.evidence.is_relative_to(self.worker):
            raise ValueError('Exchange reservations require a separate controller directory')
        proof = Path(config['qualification_file']).resolve()
        if sha(proof) != config['qualification_sha256']:
            raise ValueError('Qualification evidence changed')
        qualified = json.loads(proof.read_bytes())
        fingerprint = config.get('policy_fingerprint', '')
        if not re.fullmatch('[0-9a-f]{64}', fingerprint):
            raise ValueError('A qualified policy fingerprint is required')
        bridge_sha = sha(Path(__file__).with_name('bridge.py'))
        if (qualified.get('status') != 'physical_checks_passed' or
                qualified.get('native_reasoning') is not True or
                qualified.get('policy_fingerprint') != fingerprint or
                qualified.get('bridge_sha256') != bridge_sha):
            raise ValueError('Physical qualification does not match this policy and bridge')
        self.proof, self.proof_sha = proof, config['qualification_sha256']
        self.evidence.mkdir(parents=True, exist_ok=True)
        frozen = self.evidence / 'registration.json'
        try:
            exclusive(frozen, self.blob)
        except FileExistsError:
            if frozen.read_bytes() != self.blob:
                raise ValueError('Development registration changed after exchange creation')

    def check_frozen(self):
        if self.registration.read_bytes() != self.blob or sha(self.proof) != self.proof_sha:
            raise ValueError('Frozen exchange inputs changed')
        if sha(Path(__file__).with_name('bridge.py')) != json.loads(self.proof.read_bytes())['bridge_sha256']:
            raise ValueError('Bridge changed after qualification')

    def check_binding(self, request, binding):
        if request.get('episode_id') != self.config['episode_id']:
            raise ValueError('Request belongs to another registered episode')
        if (binding.get('episode_id') != request['episode_id'] or
                type(request.get('sequence')) is not int or
                type(binding.get('sequence')) is not int or
                binding['sequence'] != request['sequence']):
            raise ValueError('Request belongs to a stale observation')

    def submit(self, staged_request, binding):
        """Reserve before publication; an ambiguous publication is never replayed."""
        self.check_frozen()
        staged_request = Path(staged_request).resolve()
        if not staged_request.is_relative_to(self.input_root / 'captures'):
            raise ValueError('Request must come from this worker causal capture directory')
        raw = staged_request.read_bytes()
        if len(raw) > 65536:
            raise ValueError('Request metadata exceeds its bound')
        request = json.loads(raw)
        identifier = request.get('request_id', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', identifier):
            raise ValueError('Invalid request identifier')
        if request.get('fixture_only') is not False or request.get('source') != 'official_allowed_observation':
            raise ValueError('Only captured allowed observations may be submitted')
        self.check_binding(request, binding)
        history = request.get('history')
        if not isinstance(history, list) or not 1 <= len(history) <= 6:
            raise ValueError('A bounded causal history is required')
        for row in history:
            if sha(contained(self.input_root, row['path'])) != row['sha256']:
                raise ValueError('Captured observation changed before publication')
        request_sha = hashlib.sha256(raw).hexdigest()
        reservations = self.evidence / 'requests'
        reservations.mkdir(exist_ok=True)
        record_path = reservations / (identifier + '.json')
        published = self.input_root / 'requests' / (identifier + '.json')
        with (self.evidence / 'exchange.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.check_frozen()
            if record_path.exists():
                old = json.loads(record_path.read_bytes())
                if old['request_sha256'] != request_sha:
                    raise ValueError('Request identifier reused with another input')
                if not published.is_file() or sha(published) != request_sha:
                    raise RuntimeError('Publication is unconfirmed; inspect evidence without resubmitting')
                return {'request_id': identifier, 'request_sha256': request_sha, 'reused_publication': True}
            if len(list(reservations.glob('*.json'))) >= self.config['model_call_limit']:
                raise RuntimeError('Development model-call budget exhausted')
            # This reservation remains counted even if publication is interrupted.
            exclusive(record_path, encoded({'request_id': identifier, 'request_sha256': request_sha,
                'episode_id': request['episode_id'], 'sequence': request['sequence'],
                'registration_sha256': hashlib.sha256(self.blob).hexdigest()}))
            exclusive(published, raw)
        return {'request_id': identifier, 'request_sha256': request_sha, 'reused_publication': False}

    def collect(self, identifier, binding, proprio):
        """Validate the worker receipt and publish one executable candidate file."""
        self.check_frozen()
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', identifier):
            raise ValueError('Invalid request identifier')
        reservation = json.loads((self.evidence / 'requests' / (identifier + '.json')).read_bytes())
        self.check_binding(reservation, binding)
        ledger = self.worker / 'calls' / (identifier + '.json')
        if not ledger.exists():
            return {'status': 'pending', 'request_id': identifier}
        receipt = json.loads(ledger.read_bytes())
        if receipt.get('request_sha256') != reservation['request_sha256']:
            raise ValueError('Worker receipt belongs to another input')
        if receipt.get('status') == 'reserved':
            return {'status': 'pending', 'request_id': identifier}
        if receipt.get('status') != 'completed':
            raise RuntimeError('Model request is incomplete or failed; inspect without replay')
        result_file = contained(self.worker, receipt['result_file'])
        if sha(result_file) != receipt['result_sha256']:
            raise ValueError('Completed worker result changed')
        result = json.loads(result_file.read_bytes())
        if result.get('request_id') != identifier or result.get('request_sha256') != reservation['request_sha256']:
            raise ValueError('Result identity differs from the request reservation')
        self.check_binding(result.get('binding', {}), binding)
        if result.get('native_reasoning') is not True or not isinstance(result.get('native_reasoning_text'), str):
            raise ValueError('Development mode requires native reasoning text with continuous actions')
        if result.get('status') == 'candidate_rejected':
            return {'status': 'candidate_rejected', 'request_id': identifier,
                    'rejection': result.get('rejection', 'Frozen action bounds rejected the candidate')}
        if result.get('status') != 'candidate':
            raise ValueError('Unknown model result status')
        proposal = result['proposal']
        if proposal.get('proposal_id') != identifier or proposal.get('policy_fingerprint') != self.config['policy_fingerprint']:
            raise ValueError('Candidate policy or proposal identity changed')
        joint_tool_arguments(proposal, proprio, binding['episode_id'], binding['sequence'])
        destination = self.workspace / 'perception' / 'g05_proposals' / (identifier + '.json')
        blob = encoded(proposal)
        try:
            exclusive(destination, blob)
        except FileExistsError:
            if destination.read_bytes() != blob:
                raise ValueError('Published proposal content changed')
        return {'status': 'candidate', 'request_id': identifier, 'proposal_file': str(destination),
                'proposal_sha256': hashlib.sha256(blob).hexdigest(), 'physical_execution': False}
