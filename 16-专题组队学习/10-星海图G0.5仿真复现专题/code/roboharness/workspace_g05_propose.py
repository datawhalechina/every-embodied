"""Request one observation-bound G0.5 candidate for a current subtask.

Arguments carry the subtask, reproducible seed and candidate index. Each
unique set of arguments reserves a separate inference from the finite budget.
A pending request may be collected again with the same arguments; this tool
never sends physical commands.
"""

NAME='g05_propose'
SCHEMA={'type':'object','properties':{'task':{'type':'string','minLength':1,'maxLength':8192},
    'seed':{'type':'integer','minimum':0,'maximum':4294967295},
    'candidate_index':{'type':'integer','minimum':0,'maximum':2}},
    'required':['task','seed','candidate_index'],'additionalProperties':False}


if __name__=='__main__':
    raise RuntimeError('g05_propose requires the registered hybrid host')
