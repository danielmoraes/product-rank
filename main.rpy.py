from regression import *
import rpy
from rpy import r
#import rpy2.robjects

#r = rpy2.robjects.r

data_version = '2012-10-05/3/'
data_file = 'std_offer_specs_1_refined_1_fix.csv'

r.setwd('/home/daniel/Dropbox/academic/unicamp.cs.msc/project/spider.data/working/' +
        data_version)

r.options(stringsAsFactors = True)

reg = Regression(r['read.table'](data_file, header = True, sep = ",",
    stringsAsFactors = True))
reg.lm(1,2)
