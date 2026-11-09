# Project Status

## Task 0.1 Environment & Device Check
- status: done
- what changed: Verified .venv with torch 2.8.0, torchvision, matplotlib, numpy, pandas. Benchmarked CPU vs MPS on MLP operations.
- key numbers: CPU: 0.156s vs MPS: 0.317s for 500 MLP batch forward passes. CPU has significantly lower dispatch latency for small MLP matrix operations.
- problems / decisions: Use CPU for all runs as specified in plan.
- next: Task 0.2

## Task 0.2 Freeze & Project Structure
- status: done
- what changed: cl.py frozen. Copied cl.py to cl2.py. Created folders lm/, outputs/phase1/, outputs/phase2/, outputs/phase3/, outputs/phase4/, outputs/phase5/, outputs/failed/, logs/, configs/. Created STATUS.md. Verified acceptance with `python cl2.py --seeds 1 --epochs 1`.
- key numbers: Smoke test user time: 15.87s. Smoke test table: finetune ACC 88.0%, replay ACC 92.9%, lora ACC 90.1%, gate:mag ACC 89.7%, gate:conf ACC 32.4%, ewc ACC 9.8% (diverged).
- problems / decisions: cl.py is untouched. All Phase 1+ code will be developed in cl2.py.
- next: Task 1.1
