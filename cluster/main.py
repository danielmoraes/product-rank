# adding my personal library to the system path
import sys
sys.path.insert(0, '/home/dmoraes/lib/python/')
sys.path.insert(1, '/home/dmoraes/usr/lib/')

# adding the R library path to the os environment
import os
os.environ['LD_LIBRARY'] = '${LD_LIBRARY_PATH}:/usr/lib/R/lib'

# PricePrediction strategy
from price_prediction import PricePrediction

# python to R interface
import rpy2.robjects as robjects
r = robjects.r

# numpy lib
import numpy as np

class Main:
    "Main"

    def __init__(self, rankStrategy, technique, techParams, workingDir, dataFile):
        """ Class constructor """
       
        self.rankStrategy = rankStrategy
        self.technique = technique
        self.techParams = techParams

        # setting the python and R working directory
        self.workingDir = workingDir
        r.setwd(workingDir)

        # cluster data path
        self.rDataFile = r['read.table'](dataFile, header = True, sep = ",", quote='',
                stringsAsFactors = True)

    def run(self):
        if self.rankStrategy == 'price_prediction':
            if self.technique == 'lm':
                maxDegree = self.techParams[0]
                self.predictLM(maxDegree)
            elif self.technique == 'svr':
                resDirName = self.techParams[0]
                crossType = self.techParams[1]
                self.predictSVR(resDirName, crossType)
            elif self.technique == 'genetic':
                modelDegree = self.techParams[0]
                self.predictGenetic(modelDegree)

    def predictLM(self, maxDegree):
        """ Predict prices using polynomial regression """
        
        pricePrediction = PricePrediction(self.rDataFile, self.workingDir)

        reg_results = {}
        for i in range(1,maxDegree+1):
            fit, cross_val = pricePrediction.lm(i)

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

        for res in range(1,len(reg_results)+1):
            total = 0
            for i in range(10):
                total += reg_results[res][i]["cross"]
            mean = total/10

            print '%1.0f: %.5f' % (res, mean,)

    def predictSVR(self, resDirName, crossType):
        """ Predict prices using support vector regression """
        
        pricePrediction = PricePrediction(self.rDataFile, self.workingDir)
        pricePrediction.svr(resDirName, crossType)

    def predictGenetic(self, modelDegree):
        """ Predict prices using genetic algorithm """

        pricePrediction = PricePrediction(self.rDataFile, self.workingDir)
        pricePrediction.genetic(modelDegree)

rankStrategy = 'price_prediction'
technique = 'svr'
techParams = ['svr_hr', '2fold']
#technique = 'lm'
#techParams = [5]
workingDir = '/home/dmoraes/datasets/1/'
dataFile = 'std_offer_specs_1_refined_1_no-cpu-model_str.csv'

main = Main(rankStrategy, technique, techParams, workingDir, dataFile)
main.run()

