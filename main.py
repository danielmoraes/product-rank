from regression import *
import rpy2.robjects as robjects

r = robjects.r

data_version = '2012-10-05/3/'
data_file = 'std_offer_specs_1_refined_1_fix.csv'

r.setwd('/home/daniel/Dropbox/academic/unicamp.cs.msc/project/spider.data/working/' +
        data_version)

reg = Regression(r['read.table'](data_file, header = True, sep = ",",
    stringsAsFactors = True))
reg.lm(1,10)
