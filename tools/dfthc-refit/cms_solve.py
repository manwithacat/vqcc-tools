#!/usr/bin/env python3
# Copyright 2026 James Barlow
# SPDX-License-Identifier: Apache-2.0
# AI-assisted (Claude, Anthropic), under the repository owner's direction.
"""Solve a DIMACS file (with CryptoMiniSat 'x' XOR lines) through pycryptosat; print SAT-competition output.

    "$CMS_PYTHON" cms_solve.py FILE.cnf TIME_LIMIT_S [THREADS]

Exit 10 SAT ('s SATISFIABLE' and 'v' lines), 20 UNSAT, 0 unknown ('s UNKNOWN'). No proof is written (pycryptosat
has no proof interface): an UNSAT from this script is a solver claim, proof unchecked. CMS_PYTHON is any interpreter with pycryptosat.
"""
import sys

import pycryptosat


def main():
    path, limit = sys.argv[1], float(sys.argv[2])
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    s = pycryptosat.Solver(time_limit=limit, threads=threads)
    nx = nc = 0
    with open(path) as f:
        for line in f:
            if not line.strip() or line[0] in "cp":
                continue
            if line[0] == "x":
                lits = [int(t) for t in line[1:].split()][:-1]
                rhs = True
                for l_ in lits:
                    if l_ < 0:
                        rhs = not rhs
                s.add_xor_clause([abs(l_) for l_ in lits], rhs)
                nx += 1
            else:
                s.add_clause([int(t) for t in line.split()][:-1])
                nc += 1
    print(f"c pycryptosat {pycryptosat.__version__}: {nc} clauses, {nx} xor constraints", flush=True)
    sat, model = s.solve()
    if sat is True:
        print("s SATISFIABLE")
        print("v " + " ".join(str(i if v else -i) for i, v in enumerate(model) if i and v is not None) + " 0")
        sys.exit(10)
    if sat is False:
        print("s UNSATISFIABLE")
        sys.exit(20)
    print("s UNKNOWN")
    sys.exit(0)


if __name__ == "__main__":
    main()
