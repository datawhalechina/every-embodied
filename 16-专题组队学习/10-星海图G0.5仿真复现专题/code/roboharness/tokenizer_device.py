"""Place the separate action-codec backend beside the inference tensors."""


def align_action_tokenizer(policy,device):
    import torch
    tokenizer=policy.processor.action_tokenizer
    backend=getattr(tokenizer,'action_tokenizer',None)
    if not isinstance(backend,torch.nn.Module):
        raise TypeError('Official action tokenizer backend must be a torch module')
    tokenizer.to(device)
    expected=torch.device(device)
    tensors=list(backend.parameters())+list(backend.buffers())
    if not tensors or any(value.device!=expected for value in tensors):
        raise RuntimeError('Action codec backend did not move to the inference device')
    backend.eval()
    return {'device':str(expected),'parameter_tensors':len(list(backend.parameters())),
        'buffer_tensors':len(list(backend.buffers())),
        'floating_dtypes':sorted({str(value.dtype) for value in tensors if value.is_floating_point()})}
