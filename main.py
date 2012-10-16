from regression_cross import *
import rpy2.robjects as robjects
import numpy as np

r = robjects.r

data_version = '2012-10-05/4/'
data_file = 'std_offer_specs_1_refined_1_no-cpu-model.csv'
working_dir = '/home/daniel/Dropbox/academic/unicamp.cs.msc/project/spider.data/working/' + data_version

# setting the python and R working directory
r.setwd(working_dir)

def run_lm(reg):
    # linear and polynomial regression
    max_degree = 10
    reg_results = {}
    for i in range(1,max_degree+1):
        fit, cross_val = reg.lm(i)

        reg_results[i] = [None] * 10
        
        for j in range(10):
            residuals = r['summary'](fit)[2]
            coefficients = r['summary'](fit)[3]
            aliased = r['summary'](fit)[4]
            sigma  = r['summary'](fit)[5][0]
            df = r['summary'](fit)[6][0]
            r_squared = r['summary'](fit)[7][0]
            adj_r_squared = r['summary'](fit)[8][0]
            cross = np.mean(np.absolute(np.subtract(cross_val[j]['pred_prices'], 
                            cross_val[j]['real_prices'])) / cross_val[j]['real_prices'])

            reg_results[i][j] = {
                'residuals': residuals, 
                'coefficients': coefficients,
                'aliased': aliased,
                'sigma': sigma,
                'df': df,
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'cross': cross
            }

    for res in reg_results:
        total = 0
        for i in range(10):
            total += reg_results[res][i]["cross"]
        mean = total/10

        print str(res) + ': ' + str(mean)

def run_svr(reg):
    # support vector regression
    reg.svr()

# Regression instance
reg = Regression(r['read.table'](data_file, header = True, sep = ",",
    stringsAsFactors = True), working_dir)

run_svr(reg)

