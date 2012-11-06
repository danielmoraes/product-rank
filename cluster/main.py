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

    def __init__(self, rankStrategy, technique, techParams, workingDir,
            dataFile, valIds = []):
        """ Class constructor """
       
        self.rankStrategy = rankStrategy
        self.technique = technique
        self.techParams = techParams

        # setting the python and R working directory
        self.workingDir = workingDir
        r.setwd(workingDir)

        self.rDataFile = r['read.table'](dataFile, header = True, sep = ",", quote='',
                stringsAsFactors = True)

        self.valIds = valIds

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
        
        pricePrediction = PricePrediction(self.rDataFile, self.workingDir,
                self.valIds)

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

        pricePrediction = PricePrediction(self.rDataFile, self.workingDir,
                self.valIds)
        pricePrediction.svr(resDirName, crossType)

    def predictGenetic(self, modelDegree):
        """ Predict prices using genetic algorithm """

        pricePrediction = PricePrediction(self.rDataFile, self.workingDir,
                self.valIds)
        pricePrediction.genetic(modelDegree)

rankStrategy = 'price_prediction'
technique = 'svr'
techParams = ['svr_hr', '2fold']
#technique = 'lm'
#techParams = [5]
workingDir = '/home/dmoraes/datasets/1/'
dataFile = 'std_offer_specs_1_refined_1_no-cpu-model_str.csv'

valIds = ["10042042071473919540", "10115557741499455579",
          "10198992337604092255", "10660778921539065951", "10836067036626445968",
          "10840594361218801501", "11075953045429478446", "1115770551400009487",
          "11197160959360408895", "11454960459781727628", "11530282848970745391",
          "11540055609545135557", "11544889873844603832", "11644173636197049081",
          "121393340552408620", "12454419868848728357", "12658661573620019110",
          "1269077004618136411", "13009415889885870525", "13065498101602502269",
          "13077806425412745049", "1326601974741727223", "13340114743452940308",
          "13486961177026390118", "13578758997251769242", "13640541194109558825",
          "13650586586314627826", "13795479952677497758", "13999952642140301421",
          "14351366332599145086", "14667683723488806969", "14887574878643139909",
          "15514096673674662274", "1556139017108071009", "15630495468350842421",
          "15676465151754481270", "15700956778227868322", "15832460680890497046",
          "15956991559213678896", "16062413441531954747", "16133277327554207772",
          "16296489504445822662", "16339705762773060111", "16881539705098457661",
          "16960399523252868813", "17003039771047908208", "17027936431095655851",
          "17193588718582764562", "17795348434156827663", "17818785705188559565",
          "17828380218855946271", "17891827491669968268", "17970471663419943956",
          "18187469172894655333", "18259807734269833476", "2033255983674970214",
          "2260755193876131907", "229831732984625798", "2400915835830124098",
          "2812302064763831962", "3357274925576012669", "3636511060856514098",
          "3639847870770156967", "3798501454412744418", "3885712002505993349",
          "3905643410280481454", "4006556787408523793", "4201105441247825184",
          "4230688520128215935", "4488190483628909777", "4492264831939922519",
          "4519400601461371159", "4526062704021156700", "4951696076046304705",
          "5238296034497477044", "5667355130474897285", "5692045396867215843",
          "5864663906864527368", "591996572399293829", "6255741892431163040",
          "6349231821864840119", "6364219505840161130", "6849908377245687254",
          "6954492747098322828", "7320762352381137486", "7622575789119914178",
          "777415161209995588", "787711980568006406", "7906825960605741597",
          "8023409695905298613", "8194206286320205104", "8288152357428885260",
          "8314969399756073668", "8667841844491088304", "8781566056153779038",
          "964258365948286848", "9768395304785573367", "9771940656370572981",
          "9815281397449858316", "9819797841580929529"]

main = Main(rankStrategy, technique, techParams, workingDir, dataFile, valIds)
main.run()

