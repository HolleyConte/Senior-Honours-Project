from cosmosis.datablock import option_section



def setup(options):
    section = options.get_string(option_section, "section", default="likelihood")
    name = options.get_string(option_section, "name", default="2pt_like")
    temperature = options.get_double(option_section, "temperature", default=1.0)

    config = [section, name, temperature]
    return config

def execute(block, config):
    section, name, temperature = config

    # 1. Get the current log-likelihood value from the block using the section and name as keys
    log_likelihood = block[section, name]


    # 2. Modify the log-likelihood as we want
    heated_log_likelihood = log_likelihood / temperature


    # 3. Save it back to the block
    block[section, name] = heated_log_likelihood


    return 0


def cleanup(config):
    pass