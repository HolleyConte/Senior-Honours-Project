import numpy as np
import glob
import cosmosis

# 1. initialize the likelihood pipeline
p = cosmosis.LikelihoodPipeline("params.ini")

######changing index dynamically:
#p = cosmosis.LikelihoodPipeline("params.ini", override={("load_nz", "index"): "-1" })
#####

files = glob.glob('/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/all_n_z_realizations/*.txt')
results = []

# 2. loop through each realization n(z) index file
for file in files:
    data = np.loadtxt(file)

    weights = data[:, 6]  # 7th column for log_weight
    # Clip weights to avoid overflow
    clipped_weights = np.clip(weights, -500, 500)
    max_weight = np.max(clipped_weights)
    total_weight = np.sum(np.exp(clipped_weights - max_weight))
    probabilities = np.exp(clipped_weights - max_weight) / total_weight
    
    # draw 10 random samples based on their weights
    sample_indices = np.random.choice(len(data), size=10, p=probabilities)
    
    # run posterior processing for each selected sample
    for index in sample_indices:
        sample = data[index]

        # exclude the last 3 columns
        input_vector = sample[:-3]

        # take that row of data (random sample) and pass it through the pipeline “p.posterior”
        log_posterior, extra_output = p.posterior(input_vector)
        
        # append results
        results.append(np.array(np.hstack((sample, log_posterior))))

# save results to a .txt file with a header
output_file = '/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/sampled_results1.txt'
header = 'col1 col2 col3 col4 col5 col6 col7 col8 col9 col10 col11 col12 log_posterior'
np.savetxt(output_file, results, fmt='%.6f', header=header, comments='')
