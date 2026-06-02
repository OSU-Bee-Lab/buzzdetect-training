# deeper-head

## Hypothesis
The current classifier is a single Dense layer (logistic regression) over YAMNet's 1024-dim embeddings. YAMNet was trained on AudioSet and its embeddings encode a wide variety of audio concepts — buzz-relevant features are almost certainly entangled with irrelevant ones in a nonlinear way. Adding one hidden layer (Dense 128, ReLU) before the output should allow the model to learn nonlinear combinations and improve sensitivity without sacrificing precision.

## Evidence
- Linear probing on pretrained audio embeddings is a common starting point but rarely optimal; a shallow MLP typically yields gains of 2–5% on detection tasks (e.g. HEAR benchmark results).
- The current head is architecturally the simplest possible; there is no prior experiment ruling out more capacity.
