#!/bin/sh
# Card t_a696ce02 — commit helper in the private clone (explicit paths only).
export GIT_CONFIG_GLOBAL=${GIT_CONFIG_GLOBAL:-/root/.gitconfig}
cd /work/t_a696ce02/repo || exit 1
echo "--- status ---"
git status --short
if [ "$1" = "commit" ]; then
    shift
    msg=$1; shift
    git add "$@" || exit 2
    git -c user.name=Rybens92 -c user.email=rybens92@gmail.com commit -q -m "$msg" || exit 3
    git log --oneline -1
fi
