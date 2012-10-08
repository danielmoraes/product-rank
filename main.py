from regression import *
import rpy2.robjects as robjects
import numpy as np

r = robjects.r

data_version = '2012-10-05/3/'
data_file = 'std_offer_specs_1_refined_1_fix.csv'

r.setwd('/home/daniel/Dropbox/academic/unicamp.cs.msc/project/spider.data/working/' +
        data_version)

reg = Regression(r['read.table'](data_file, header = True, sep = ",", stringsAsFactors = True))

'''

max_degree = 3
reg_results = {}
for i in range(1,max_degree+1):
    fit, cross_val = reg.lm(i)
    
    residuals       = r['summary'](fit)[2]
    coefficients    = r['summary'](fit)[3]
    aliased         = r['summary'](fit)[4]
    sigma           = r['summary'](fit)[5][0]
    df              = r['summary'](fit)[6][0]
    r_squared       = r['summary'](fit)[7][0]
    adj_r_squared   = r['summary'](fit)[8][0]
    cross           = np.mean(np.absolute( np.subtract(cross_val['pred_prices'], cross_val['real_prices'])  ) / cross_val['real_prices'])

    reg_results[i] = {
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
    print str(res) + ': ' + str(reg_results[res]["cross"])

'''

reg.svr()
