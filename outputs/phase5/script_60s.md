# 60-Second Spoken Poster Script

**Estimated Word Count:** 142 words  
**Speaking Pace:** ~140 words/minute (approx. 60 seconds)  

---

*(Begin speaking)*

"Catastrophic forgetting is a major bottleneck in AI: train a network on new tasks, and it aggressively overwrites old knowledge. 

In this work, we tested whether a buffer-free **Gradient-Conflict Gate** could protect past weights without storing data. 

Under an honest multi-seed benchmark on Permuted and Split-MNIST, our first discovery was a severe failure: on 10 sequential tasks, the original conflict gate collapsed to **41.9% accuracy**, falling **36% below plain fine-tuning**. 

Parameter telemetry supports this mechanism: blocking opposing updates creates an **asymmetric ratchet** that doubles weight norms and causes runaway drift. 

By designing an anti-drift gate that limits expansion from starting weights alongside hybrid damping, we eliminated the runaway. On 10 tasks, our repaired gates reached **83.8% and 83.2% accuracy**, statistically outperforming plain fine-tuning by over **5%** without storing any historical samples. 

Honest science turned a failed idea into a robust, working mechanism."

*(End speaking)*
