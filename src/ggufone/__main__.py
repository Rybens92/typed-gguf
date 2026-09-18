"""`python -m ggufone` entry point -> `ggufone.cli.run` (the process entry point, card t_97f1bc93).

`run`, not `main`: the command's own code is this process's exit status, and a process that has a
bundle loaded ends itself instead of handing the shell whatever a third-party destructor does at
interpreter exit (`ggufone.cli.run` carries the evidence).
"""

from ggufone.cli import run

if __name__ == "__main__":  # pragma: no cover
    run()
