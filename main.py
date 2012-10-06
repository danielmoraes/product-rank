import regression
import rpy
from rpy import r

data_version = '2012-10-05/3/'
data_file = 'std_offer_specs_1_refined_1_fix.csv'

r.setwd('/home/daniel/Dropbox/academic/unicamp.cs.msc/project/spider.data/working/' +
        data_version)

reg = Regression(r.read_csv(data_file))
