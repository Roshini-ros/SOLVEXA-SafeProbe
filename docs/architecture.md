# Architecture

```text
Synthetic scenario
      |
      v
Motor signal model ----> Sensor observations
                              |
                              v
                  Multi-fault hypothesis scoring
                              |
                              v
                     Posterior distribution
                              |
                              v
              Cost/risk-constrained test selector
                              |
                              v
                 Virtual test result / update loop
```

The implementation uses an intentionally small state space: three binary fault variables yield eight possible fault combinations. This makes the reasoning visible and the demo reproducible.
