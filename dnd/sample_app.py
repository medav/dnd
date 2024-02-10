import torch
import dnd

net = torch.nn.Sequential(*[
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10),
    torch.nn.Softmax(dim=1)
]).to(dnd.env.dtype).to(dnd.env.dev)

x = torch.randn(dnd.env.bs, 512, dtype=dnd.env.dtype, device=dnd.env.dev)

def roi(x):
    with dnd.trace_region('forward'):
        y = net(x).sum()

    with dnd.trace_region('backward'):
        y.backward()

    return y

dnd.profile(roi, x)

