import numpy as np
import glob
import cosmosis



# 1. initialize the likelihood pipeline
p = cosmosis.LikelihoodPipeline("params.ini")

files = glob.glob('/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/all_n_z_realizations/*.txt')
results = []

# 2. loop through each realization n(z) index file
for file in files:
    data = np.loadtxt(file)

    weights = data[:, 6]  # 7th column for log_weight
    total_weight = np.sum(np.exp(weights))
    probabilities = np.exp(weights) / total_weight
    
    # draw 10 random samples based on their weights
    sample_indices = np.random.choice(len(data), size=10, p=probabilities)
    
    # run posterior processing for each selected sample
    for index in sample_indices:
        sample = data[index]

        # exclude the last 3 columns
        input_vector = sample[:-3]

        # take that row of data (random sample) and pass it through the pipeline “p.posterior”
        log_posterior = p.posterior(input_vector)
        
        # append results
        results.append(np.array(np.hstack((sample, log_posterior)), dtype=object))

# save results to a .txt file
output_file = '/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/sampled_results.txt'
np.savetxt(output_file, results, fmt='%.6f')