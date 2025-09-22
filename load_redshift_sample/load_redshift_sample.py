import h5py
from cosmosis.datablock import option_section
import os

def setup(options):
    # Read some options from the ini file:
    # - the name of the HDF file we will read
    # - whether we want the source or lens n(z) from the file
    # - the index number of the sample to use, or -1 for the average
    filename = options.get_string(option_section, "filename")
    source_or_lens = options.get_string(option_section, "source_or_lens", default="source")
    index = options.get_int(option_section, "index", default=-1)

    # There are only two sensible options for source_or_lens
    if source_or_lens not in ["source", "lens"]:
        raise ValueError("source_or_lens must be 'source' or 'lens'")
    
    if not os.path.isfile(filename):
        raise ValueError("You need to download the data file as noted in the readme and put it in the data directory")

    # Open the file so we can read it using the h5py library
    with h5py.File(filename, 'r') as f:
        # HDF5 files are structured into groups and data sets with a path,
        # like files on disc. These are where the data are stored in this file.
        if index == -1:
            data = f[f"{source_or_lens}/average"][:]
        else:
            data = f[f"{source_or_lens}/data"][index, :]

        # The redshift grid is the same for both source and lens samples
        z = f["/meta/z_grid"][:]

    # Return a config dictionary to pass to execute later
    config = {
        "z": z,
        "data": data
    }

    return config


def execute(block, config):
    # Get the things out of the config dictionary
    z = config["z"]
    data = config["data"]

    # Save them to the datablock in the right place
    block["nz_source", "z"] = z
    block["nz_source", "nbin"] = data.shape[0]
    for i in range(data.shape[0]):
        block["nz_source", f"bin_{i+1}"] = data[i, :]

    # Returning zero means success
    return 0
