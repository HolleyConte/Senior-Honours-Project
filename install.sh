#!/usr/bin/env bash
set -e
set -x

#check if running on mac or linux
UNAME=$(uname)
if [ "$UNAME" == "Linux" ]; then
    wget -O Miniforge3.sh  https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-Linux-x86_64.sh
elif [ "$UNAME" == "Darwin" ]; then
    wget -O Miniforge3.sh  https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-MacOSX-arm64.sh
else
    echo "Unsupported OS: $UNAME"
    exit 1
fi

# Make the file we just downloaded executable so we can run it
chmod +x Miniforge3.sh

# Run the Miniforge installer, which will install a new python environment
# for us in the "env" directory. -p means "prefix" (i.e. where to install it)
# -b means "batch" (i.e. don't ask any questions, just install it)
./Miniforge3.sh -b -p ./env

# activate the environment we just created
source ./env/bin/activate

# install the things we need
mamba install -y cosmosis cosmosis-build-standard-library cosmopower

# build cosmosis
source ./env/bin/activate
source cosmosis-configure
cosmosis-build-standard-library spk

echo
echo "Installation complete! You don't need to run the installer script again."
echo "To use the environment, run these two commands:"
echo "source ./env/bin/activate"
echo "source cosmosis-configure"
echo
