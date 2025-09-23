#!/usr/bin/env bash
set -e
set -x

# We download a package called "Miniforge" which lets us build an isolated
# python environment in a local directory, so it won't interfere with anything
# that you already have installed.
UNAME=$(uname)
if [ "$UNAME" == "Linux" ]; then
    URL="https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-Linux-x86_64.sh"
elif [ "$UNAME" == "Darwin" ]; then
    URL="https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-MacOSX-arm64.sh"
else
    echo "Unsupported OS: $UNAME"
    exit 1
fi

# The curl command downloads a file from the internet. The -o option
# lets us specify the name of the file we want to save it as.
curl -L -o Miniforge3.sh $URL

# Make the file we just downloaded executable so we can run it
chmod +x Miniforge3.sh

# Run the Miniforge installer, which will install a new python environment
# for us in the "env" directory. -p means "prefix" (i.e. where to install it)
# -b means "batch" (i.e. don't ask any questions, just install it)
./Miniforge3.sh -b -p ./env

# activate the environment we just created
source ./env/bin/activate

# install the things we need.
mamba install -y cosmosis=3.23 cosmosis-build-standard-library cosmopower

# build cosmosis
source ./env/bin/activate
source cosmosis-configure

# Build the standard library, using the "spk" branch which
# includes the cosmopower emulator. This is a faster version
# of one of our standard pipelines.
cosmosis-build-standard-library spk

echo
echo "Installation complete! You don't need to run the installer script again."
echo "To use the environment, run these two commands:"
echo "source ./env/bin/activate"
echo "source cosmosis-configure"
echo
