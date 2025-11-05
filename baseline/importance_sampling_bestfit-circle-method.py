import numpy as np
import glob
import cosmosis



# 1. load the big schmear+high temp chain
schmear_data = np.loadtxt('/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/schmear_0.2_AND_temp_20.txt')
schmear_weights = schmear_data[:, 16]  # 17th column for log weights


# 1.5. initialize the "schmear+temp" likelihood pipeline
schmear_pipeline = cosmosis.LikelihoodPipeline("params.ini")

files = glob.glob('/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/all_n_z_realizations/*.txt')
results = []

# 2. loop through each realization n(z) index file
for file in files:
    index_data = np.loadtxt(file)

    # 2.1 create a new pipeline object for the index
    index_pipeline = cosmosis.LikelihoodPipeline("params.ini", override={("load_nz", "index"): -1})

    # 2.2 load the index chain file for that index
    index_weights = index_data[:, 6]  # 7th column for log weights


    # 2.3 find the maximum likelihood sample point to do the "circle method"
    max_likelihood_index = np.argmax(index_weights)
    max_likelihood_sample = cut_schmear_data[max_likelihood_index]

    # 2.4 calculate the mean and standard deviation for the circle around the max point
    mean_point = max_likelihood_sample[:-3]  # Exclude the last 3 columns
    radius = 0.1  # radius of the circle

    # 2.5 create a mask for the circle around the max point
    distances = np.linalg.norm(cut_schmear_data[:, :-3] - mean_point, axis=1)
    circle_mask = distances <= radius

    # 2.6 cut the schmear+T chain to be only within the circle
    cut_schmear_data = cut_schmear_data[circle_mask]


    # 2.7 run the index pipeline on the subset of schmear+T and get posterior
    for index in cut_schmear_data:
        sample = index_data[index]

        # exclude the last 3 columns
        input_vector = sample[:-3]

        # take that row of data and pass it through the pipeline “index_pipeline.posterior”
        log_posterior, extra_output = index_pipeline.posterior(input_vector)

        # save new points, old weight, new posterior for each index
        results.append(np.array(np.hstack((sample, index_weights, log_posterior))))


# 3. save results to a .txt file with a header
output_file = '/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/sampled_results.txt'
header = 'col1 col2 col3 col4 col5 col6 col7 col8 col9 col10 col11 col12 old_weight log_posterior'
np.savetxt(output_file, results, fmt='%.6f', header=header, comments='')


# Save the "circle cut" schmear+T chain to a .txt file
circle_cut_file = '/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/circle_cut_schmear.txt'
np.savetxt(circle_cut_file, cut_schmear_data, fmt='%.6f')
