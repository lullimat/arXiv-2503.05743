# Script for the installation of idea.deploy python virtual environment
# Copyright (C) 2020-2025 Matteo Lulli (matteo.lulli@gmail.com)
# Permission to copy and modify is granted under the MIT license
# Last revised 13/6/2025

echo "Installing the idea.deploy environment for the repository of the paper:"
echo "'A note on the lattice momentum balance in the lattice Boltzmann interaction-framework'"

echo
bash -c "$(curl -fsSL https://raw.githubusercontent.com/lullimat/idea.deploy/refs/heads/master/idpy-bootstrap.sh)"
echo

# Sourcing aliases
ALIAS_SOURCE="$(cat ~/.bashrc | grep idea.deploy | grep source)"

if [ -z "${ALIAS_SOURCE}" ]; then
    echo "Error: No 'source' command for idea.deploy found in ~/.bashrc"
    exit 1
fi

# Expanding aliases as in idea.deploy
shopt -s expand_aliases
eval "${ALIAS_SOURCE}" || {
    echo "Error: Failed to source idea.deploy"
    exit 1
}

idpy-go
idpy-load
cd ./papers/
echo ${PWD}

python idpy-papers.py --repo arXiv-2503.05743
cd ./arXiv-2503.05743/
echo
echo "The environemnt idea.deploy is ready and the paper repository has been succefully cloned!"
echo
echo "Please run 'cd ${PWD}' to go to the paper repository"
echo