"""
Selective Plasticity prototype.
Benchmark: Permuted-MNIST (domain-incremental), MLP, plain SGD.
Methods:  finetune | ewc | replay | lora | gate[:mode[:beta]]
          gate modes: conflict (ours), magnitude (ablation), layer_cos (variant)
"""
import argparse, json, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets

dev = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- data
def load_mnist():
    tr = datasets.MNIST("data", train=True, download=True)
    te = datasets.MNIST("data", train=False, download=True)
    f = lambda d: ((d.data.float() / 255 - 0.1307) / 0.3081).view(len(d.data), -1)
    return f(tr).to(dev), tr.targets.to(dev), f(te).to(dev), te.targets.to(dev)


def make_perms(n_tasks, seed):
    g = torch.Generator().manual_seed(seed)
    perms = [torch.arange(784)] + [torch.randperm(784, generator=g) for _ in range(n_tasks - 1)]
    return [p.to(dev) for p in perms]


# ---------------------------------------------------------------- model
class LoRALinear(nn.Module):
    def __init__(self, i, o, n_tasks, r):
        super().__init__()
        self.base = nn.Linear(i, o)
        self.A = nn.ParameterList([nn.Parameter(torch.randn(r, i) / i ** 0.5) for _ in range(n_tasks)])
        self.B = nn.ParameterList([nn.Parameter(torch.zeros(o, r)) for _ in range(n_tasks)])
        self.t = None  # None = base only

    def forward(self, x):
        y = self.base(x)
        if self.t is not None:
            y = y + F.linear(F.linear(x, self.A[self.t]), self.B[self.t])
        return y


class MLP(nn.Module):
    def __init__(self, n_tasks=1, r=8, lora=False, h=256):
        super().__init__()
        mk = (lambda i, o: LoRALinear(i, o, n_tasks, r)) if lora else nn.Linear
        self.l1, self.l2, self.l3 = mk(784, h), mk(h, h), mk(h, 10)

    def forward(self, x):
        return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))

    def set_task(self, t):
        for m in (self.l1, self.l2, self.l3):
            if isinstance(m, LoRALinear):
                m.t = t


# ---------------------------------------------------------------- our method
class Gate:
    """Keeps d = direction old learning pushed each weight (decayed sum of per-task
    displacement). During new-task training it shrinks gradient entries whose SGD
    step would move a weight back against d."""

    def __init__(self, model, mode, beta, decay):
        self.mode, self.beta, self.decay = mode, beta, decay
        self.d = [torch.zeros_like(p) for p in model.parameters()]
        self.start = None

    def begin_task(self, model):
        self.start = [p.detach().clone() for p in model.parameters()]

    def end_task(self, model):
        for d, p, s in zip(self.d, model.parameters(), self.start):
            d.mul_(self.decay).add_(p.detach() - s)

    @torch.no_grad()
    def apply(self, model):
        for p, d in zip(model.parameters(), self.d):
            if p.grad is None or not d.any():
                continue
            g = p.grad
            # how strongly old learning moved this weight (1.0 = at or above layer average)
            mag = (d.abs() / (d.abs().mean() + 1e-12)).clamp(max=1.0)
            if self.mode == "conflict":
                conflict = (g * d > 0).float()  # step -g points against d
                g.mul_(1 - self.beta * mag * conflict)
            elif self.mode == "magnitude":
                g.mul_(1 - self.beta * mag)
            elif self.mode == "layer_cos":
                c = -(g * d).sum() / (g.norm() * d.norm() + 1e-12)  # cos(step, d)
                if c < 0:
                    g.mul_(1 - self.beta * (-c))


# ---------------------------------------------------------------- EWC helpers
def fisher_diag(model, Xp, n=500):
    fis = [torch.zeros_like(p) for p in model.parameters()]
    idx = torch.randperm(Xp.size(0), device=dev)[:n]
    for i in idx:
        model.zero_grad()
        out = model(Xp[i:i + 1])
        probs = F.softmax(out, 1)
        try:
            yi = torch.multinomial(probs, 1).squeeze(1)  # sample from model's own output
        except RuntimeError:
            break
        F.cross_entropy(out, yi).backward()
        for f, p in zip(fis, model.parameters()):
            f += p.grad.detach() ** 2 / n
    model.zero_grad()
    return fis


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, Xtes, yte, kind):
    accs = []
    for j, Xj in enumerate(Xtes):
        if kind == "lora":
            model.set_task(None if j == 0 else j)
        accs.append((model(Xj).argmax(1) == yte).float().mean().item())
    return accs


def train_task(model, opt, Xp, y, args, kind, st, gate):
    n = Xp.size(0)
    for _ in range(args.epochs):
        idx = torch.randperm(n, device=dev)
        for i in range(0, n, args.bs):
            b = idx[i:i + args.bs]
            loss = F.cross_entropy(model(Xp[b]), y[b])
            if kind == "ewc" and st["F"] is not None:
                pen = sum((f * (p - s) ** 2).sum()
                          for f, p, s in zip(st["F"], model.parameters(), st["star"]))
                loss = loss + args.ewc_lambda / 2 * pen
            if kind == "replay" and st["bx"] is not None:
                j = torch.randint(0, st["bx"].size(0), (args.bs,), device=dev)
                loss = loss + F.cross_entropy(model(st["bx"][j]), st["by"][j])
            opt.zero_grad()
            loss.backward()
            if gate is not None:
                gate.apply(model)
            opt.step()


def run(method, args, seed, data):
    set_seed(seed)
    Xtr, ytr, Xte, yte = data
    parts = method.split(":")
    kind = parts[0]
    mode = parts[1] if len(parts) > 1 else "conflict"
    beta = float(parts[2]) if len(parts) > 2 else args.beta

    perms = make_perms(args.tasks, seed)
    Xtes = [Xte[:, p] for p in perms]
    model = MLP(n_tasks=args.tasks, lora=(kind == "lora")).to(dev)
    base = evaluate(model, Xtes, yte, kind)  # untrained reference for FWT
    gate = Gate(model, mode, beta, args.decay) if kind == "gate" else None
    st = {"F": None, "star": None, "bx": None, "by": None}
    R = []

    for t, p in enumerate(perms):
        Xp = Xtr[:, p]
        if kind == "lora":
            if t > 0:
                for q in model.parameters():
                    q.requires_grad_(False)
                for m in (model.l1, model.l2, model.l3):
                    m.A[t].requires_grad_(True)
                    m.B[t].requires_grad_(True)
            model.set_task(None if t == 0 else t)
        if gate is not None:
            gate.begin_task(model)
        params = [q for q in model.parameters() if q.requires_grad]
        opt = torch.optim.SGD(params, lr=args.lr)
        train_task(model, opt, Xp, ytr, args, kind, st, gate)
        if gate is not None:
            gate.end_task(model)
        if kind == "ewc":
            fis = fisher_diag(model, Xp)
            st["F"] = fis if st["F"] is None else [a + b for a, b in zip(st["F"], fis)]
            st["star"] = [q.detach().clone() for q in model.parameters()]
        if kind == "replay":
            k = torch.randperm(Xp.size(0), device=dev)[:args.buf]
            st["bx"] = Xp[k] if st["bx"] is None else torch.cat([st["bx"], Xp[k]])
            st["by"] = ytr[k] if st["by"] is None else torch.cat([st["by"], ytr[k]])
        R.append(evaluate(model, Xtes, yte, kind))

    R = np.array(R)
    T = args.tasks
    return {
        "ACC": R[-1].mean(),
        "BWT": np.mean([R[-1, j] - R[j, j] for j in range(T - 1)]),
        "FWT": np.mean([R[j - 1, j] - base[j] for j in range(1, T)]),
        "NEW": np.mean(np.diag(R)),
        "R": R.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["finetune", "ewc", "replay", "lora", "gate:conflict:1.0", "gate:magnitude:1.0"])
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ewc_lambda", type=float, default=5000)
    ap.add_argument("--buf", type=int, default=200)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--decay", type=float, default=1.0)
    args = ap.parse_args()

    data = load_mnist()
    results = {}
    for m in args.methods:
        results[m] = [run(m, args, s, data) for s in range(args.seeds)]
        print(f"done {m}", flush=True)

    print(f"\n{'method':<22}{'ACC':>14}{'BWT':>14}{'FWT':>14}{'NEW':>14}")
    for m, rs in results.items():
        cells = []
        for k in ("ACC", "BWT", "FWT", "NEW"):
            v = np.array([r[k] for r in rs]) * 100
            cells.append(f"{v.mean():6.1f}+-{v.std():4.1f}")
        print(f"{m:<22}" + "".join(f"{c:>14}" for c in cells))

    json.dump(results, open("results.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for m, rs in results.items():
        curve = np.mean([np.array(r["R"])[:, 0] for r in rs], axis=0) * 100
        plt.plot(range(1, args.tasks + 1), curve, marker="o", label=m)
    plt.xlabel("tasks learned so far")
    plt.ylabel("accuracy on task 1 (%)")
    plt.title("Forgetting curve")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("forgetting_curve.png", dpi=200)
    print("saved results.json and forgetting_curve.png")


if __name__ == "__main__":
    main()
