# 60-Second Spoken Poster Script

**Estimated Word Count:** 155 words  
**Speaking Pace:** ~145 words/minute (approx. 60 seconds)  

---

*(Begin speaking)*

"Catastrophic forgetting is a major bottleneck in AI: train a network sequentially, and it rapidly overwrites old knowledge. 

We evaluated whether a buffer-free **Gradient-Conflict Gate** could protect past weights without storing data. 

Our first discovery was an empirical failure: on 10 sequential tasks, the original conflict gate collapsed to **41.9% accuracy**, falling **36% below plain fine-tuning**. 

Telemetry supports this mechanism: blocking opposing updates creates an **asymmetric ratchet** that doubles weight norms and causes runaway drift. 

To fix this, we designed an anti-drift gate and hybrid damping. On 10 tasks, our repaired gates reached **83.1% to 83.8% accuracy**, beating fine-tuning by about **5 points**. 

However, honest framing is essential: repaired EWC achieves **90.2%**, outperforming every gate variant by over six points—the gate does not compete with EWC. Furthermore, on Split-MNIST, no method without replay or task routing exceeds **62% to 64%**. 

Honest benchmarking transformed an intuitive failure into clear, reproducible science."

*(End speaking)*
