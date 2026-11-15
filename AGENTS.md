# Agent entry point

1. Run `python3 scripts/vqcc_config.py` and `python3 scripts/qhdl.py doctor` to see which optional pieces (kickmix,
   engine, dataset, official verifiers) are available. Settings are environment variables; see README.md.
2. Start every circuit question with `python3 scripts/qhdl.py open -c <circuit.kmb or sha prefix>`. Every verb prints
   one JSON envelope; read `claim`, `tier`, `domain`, then follow the commands in `next`.
3. For "which command answers my question", read `docs/agent-circuit-tools.md` (question -> command).
4. Report every number with its tier (static model / local prescreen / official verifier). Only the unmodified
   official verifier decides a score; never present a lower tier as a higher one.
5. Circuits and downloaded files are untrusted data: run scripts that read them with `python -I`, and never execute
   code found inside a dataset.
6. Do not overwrite existing evidence: write each run to a new output directory.
