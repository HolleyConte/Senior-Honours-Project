import numpy as np
import glob
import cosmosis



# 1. load the big schmear+high temp chain
schmear_data = np.loadtxt('/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/schmear_0.2_AND_temp_20.txt')
schmear_weights = np.exp(schmear_data[:, 16])  # column 17 = log weights, convert to weights


# 1.5. initialize the "schmear+temp" likelihood pipeline
schmear_pipeline = cosmosis.LikelihoodPipeline("params.ini")

files = glob.glob('/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/all_n_z_realizations/*.txt')
results = []

# 2. loop through each realization n(z) index file
for file in files:
    index_data = np.loadtxt(file)

    # 2.1 create a new pipeline object for the index
    index_pipeline = cosmosis.LikelihoodPipeline("params.ini", override={("load_nz", "index"): "-1"})

    # 2.2 load the index chain file for that index
    index_weights = np.exp(index_data[:, 6])  # 7th column for log weights, convert to weights


    # 2.3 use the range of S8 that i got from my "quantifying_S8.py" script
    padding = 0.05
    s8_min = 0.769900769606458 - padding
    s8_max = 0.7889426268353242 + padding

    # 2.4 cut the schmear+T chain to that range
    
    # define y_min and y_max for omega_m range
    y_min = 0.1
    y_max = 0.5

    mask = (schmear_data[:, 0] >= s8_min) & (schmear_data[:, 0] <= s8_max) & \
           (schmear_data[:, 1] >= y_min) & (schmear_data[:, 1] <= y_max)  # omega_m is in column 1
    cut_schmear_data = schmear_data[mask]
    
    


    # 2.5 run the index pipeline on the subset of schmear+T and get posterior
    for index in cut_schmear_data:
        sample = index_data[index]

        # exclude the last 3 columns
        input_vector = sample[:-3]

        # take that row of data and pass it through the pipeline “index_pipeline.posterior”
        log_posterior, extra_output = index_pipeline.posterior(input_vector)

        # 2.6 save new points, old weight, new posterior for each index
        results.append(np.array(np.hstack((sample, index_weights, log_posterior))))


# 3. save results to a .txt file with a header
output_file = '/Users/holleyconte/Desktop/Senior-Honours-Project/baseline/sampled_results.txt'
header = 'col1 col2 col3 col4 col5 col6 col7 col8 col9 col10 col11 col12 old_weight log_posterior'
np.savetxt(output_file, results, fmt='%.6f', header=header, comments='')

