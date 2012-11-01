import matplotlib
matplotlib.use('Agg')

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from utils import *
from libsvm2_svm import *
from libsvm2_svmutil import *
import matplotlib.pyplot as plt
import Image
import math
import os
from pyevolve import *

r = robjects.r

class Regression:
    "Regression"

    def __init__(self, data, wdir):
        self.data = data

        # rpy2 imports
        self.stats = importr('stats')
        self.base = importr('base')
        self.e1071 = importr('e1071')

        # setting the directory that the result files will be written
        self.wd = wdir
        self.wd += 'results/'
        if not os.path.exists(self.wd):
            os.makedirs(self.wd)
        os.chdir(self.wd)

    def apply_dummy_coding(self, data_frame, dc_type, degree = 1):
        y_label = 'base.price'
        
        base_price_pos = get_item_pos(r['names'](data_frame), y_label)
        
        kwargs = {}
        kwargs[y_label] = data_frame[base_price_pos]

        if dc_type != 'treatment':
            degree = 1

        for deg in range(1, degree+1):
            c_attr_pos = 0
            for attr in r['names'](data_frame):
                if attr == y_label: continue

                # ajusting the attr name in accordance with the current degree
                c_attr_label = attr
                if deg > 1: c_attr_label = c_attr_label + '.' + str(deg)

                # current attribute data
                c_attr_data = data_frame[c_attr_pos]

                if r['class'](c_attr_data)[0] != 'factor':
                    # it's a continuous variable
                    if deg == 1:
                        kwargs[c_attr_label] = list(c_attr_data)
                    else:
                        kwargs[c_attr_label] = np.power(c_attr_data, deg).tolist()
                elif attr == 'cid.product':
                    kwargs[c_attr_label] = c_attr_data 
                elif deg == 1:
                    # it's a categorical variable
                    c_attr_levels = r['levels'](c_attr_data)
                    c_attr_vector = r['as.vector'](c_attr_data)

                    if dc_type == 'treatment':
                        c_attr_contr = r['contrasts'](c_attr_data)

                    for col_idx in range(len(c_attr_vector)):
                        if dc_type == 'treatment':
                            contr_pos = get_item_pos(c_attr_levels, 
                                                     c_attr_vector[col_idx]) + 1
                            
                            for level_idx in range(1, len(c_attr_levels)):    
                                c_level_label = c_attr_label + '.' +\
                                                c_attr_levels[level_idx]
                                
                                if col_idx == 0:
                                    kwargs[c_level_label] =\
                                        [None] * len(c_attr_vector)
                                kwargs[c_level_label][col_idx] =\
                                    c_attr_contr.rx(contr_pos, level_idx)[0]
                        else:
                            level_pos = get_item_pos(c_attr_levels, 
                                 c_attr_vector[col_idx])
                            
                            for level_idx in range(len(c_attr_levels)):
                                c_level_label = c_attr_label + '.' +\
                                                c_attr_levels[level_idx]
                                
                                if col_idx == 0:
                                    kwargs[c_level_label] =\
                                        [None] * len(c_attr_vector)

                                if level_pos == level_idx: 
                                    dc_value = 1
                                else: 
                                    dc_value = 0

                                kwargs[c_level_label][col_idx] = dc_value
                c_attr_pos += 1
        
        # convert every attribute vector in a R float vector
        for label in kwargs.keys():
            if label != y_label and label != 'cid.product':
                kwargs[label] = robjects.FloatVector(kwargs[label])
        
        # convert the generated matrix to a R data frame
        dc_data_frame = r['data.frame'](**kwargs)

        return dc_data_frame

    def apply_2fold(self, data_frame, val_ids):
        cid_product_pos = get_item_pos(r['names'](data_frame), 'cid.product')
        all_cids = [item.replace('"', '') for item in
                list(r['as.vector'](data_frame[12]))]
        train_pos = [0 if cid in val_ids else 1 for cid in all_cids]

        training_rows = [i for i in range(len(all_cids))
                if train_pos[i] == 1]
        validation_rows = [i for i in range(len(all_cids))
                if train_pos[i] == 0]
        
        training_set = data_frame.rx(robjects.IntVector(training_rows), True)
        validation_set = data_frame.rx(robjects.IntVector(validation_rows), True)

        return [validation_set, training_set]

    def apply_kfold(self, data_frame, k):
        subsets = []
        
        total_nrows = len(self.data[0])
        subset_nrows = math.ceil(total_nrows/float(k))
        last_subset_nrows = total_nrows - subset_nrows * (k-1)
        
        c_start_row = -1
        for i in range(k-1):
            subsets.append(data_frame.rx(r['seq'](c_start_row + 1, 
                c_start_row + subset_nrows), True))
            c_start_row += subset_nrows

        subsets.append(data_frame.rx(r['seq'](c_start_row,
            c_start_row + last_subset_nrows), True))

        return subsets

    def normalize_dataset(self, validation_set, training_set, y_label):
        
        # mounting the data in the libsvm format
        base_price_pos = get_item_pos(training_set.names, y_label)

        y_data_training = list(training_set[base_price_pos])
        x_data_training = list(training_set)
        del x_data_training[base_price_pos]
        x_data_training = [list(item) for item in x_data_training]
        x_data_training = [list(item) for item in np.transpose(x_data_training)]

        y_data_test = list(validation_set[base_price_pos])
        x_data_test = list(validation_set)
        del x_data_test[base_price_pos]
        x_data_test = [list(item) for item in x_data_test]
        x_data_test = [list(item) for item in np.transpose(x_data_test)]

        # scaling the data
        scaled_x_data_training, x_ranges_dict =\
                normalize_data(x_data_training)
        scaled_x_data_test = normalize_data(x_data_test, x_ranges_dict)


        return [y_data_training, scaled_x_data_training, 
                y_data_test, scaled_x_data_test]

    def genetic(self, degree):
        y_label = 'base.price'
        dc_dataframe = self.apply_dummy_coding(self.data, 'treatment', degree)
        n_coefs = len(dc_dataframe) - 1

        baseprice_pos = get_item_pos(dc_dataframe.names, 'base.price')
        y_dataset = list(dc_dataframe[baseprice_pos])
        x_dataset = list(dc_dataframe)
        del x_dataset[baseprice_pos]
        x_dataset = [list(item) for item in x_dataset]
        x_dataset = [list(item) for item in np.transpose(x_dataset)]
        
        genome = G1DList.G1DList(n_coefs)
        genome.evaluator.set(self.ga_eval_func)
        genome.setParams(rangemin=0, rangemax=100, yds=y_dataset, xds=x_dataset)
        genome.mutator.set(Mutators.G1DListMutatorRealGaussian)

        ga = GSimpleGA.GSimpleGA(genome)
        ga.setMinimax(Consts.minimaxType["minimize"])
        ga.setPopulationSize(500)
        ga.setGenerations(1000)
        #ga.setInteractiveGeneration(5)

        ga.evolve(freq_stats=1)
        
        best = ga.bestIndividual()
        print best
        print "Best individual score: %.2f" % best.getRawScore()

    def ga_eval_func(self, chromosome):
        y_dataset = chromosome.internalParams["yds"]
        x_dataset = chromosome.internalParams["xds"]
        
        sq_abs_rel_errors = [None] * len(x_dataset)
        for item_idx in range(len(x_dataset)):
            pred_price = 0.0000000001
            attr_idx = 0
            for value in chromosome:
                pred_price += value * x_dataset[item_idx][attr_idx]
                attr_idx += 1

            abs_error = np.absolute(pred_price - y_dataset[item_idx])
            abs_rel_error = abs_error / y_dataset[item_idx]
            sq_abs_rel_error = np.power(abs_rel_error, 2)

            sq_abs_rel_errors[item_idx] = sq_abs_rel_error
        
        mean_sq_abs_rel_error = np.mean(sq_abs_rel_errors)
        
        return mean_sq_abs_rel_error
    
    def lm(self, degree):
        y_label = 'base.price'
        dc_data_frame = self.apply_dummy_coding(self.data, 'treatment', degree)
        
        x_labels = list(r['names'](dc_data_frame))
        del x_labels[get_item_pos(x_labels, y_label)]
        dc_data_model = (y_label + '~' + '+'.join(x_labels).replace(' ', '.'))
        
        dc_df_subsets = self.apply_kfold(dc_data_frame, 10)
        
        cross_results = []
        for i in range(10):
            validation_set = dc_df_subsets[i]
            
            training_subsets = list(dc_df_subsets)
            del training_subsets[i]

            training_set = r['rbind'](training_subsets[0], training_subsets[1],
                                      training_subsets[2], training_subsets[3],
                                      training_subsets[4], training_subsets[5],
                                      training_subsets[6], training_subsets[7],
                                      training_subsets[8])
        
            fit = self.stats.lm(r['as.formula'](dc_data_model), 
                data = training_set)
            pred_prices = r['predict'](fit, validation_set, type="response")

            real_prices = validation_set[get_item_pos(validation_set.names, 
                y_label)]
            cross_results.append({'pred_prices': pred_prices, 
                'real_prices': real_prices})

        return fit, cross_results

    def svr(self, cross_type):
        svm_wd = self.wd + 'svr_hr/'
        if not os.path.exists(svm_wd):
           os.makedirs(svm_wd)
        os.chdir(svm_wd)

        dir_files = [int(item.split('.')[0]) for item in os.listdir(svm_wd) 
                     if item.split('.')[0] != 'general']
        
        if len(dir_files) == 0:
            dir_idx = 0
        else:
            dir_idx = max(dir_files)
        
        y_label = 'base.price'
        
        # applying the coding scheme to the data
        #data_frame_training, data_frame_test = self.make_svr_data_frame()
        dc_data_frame = self.apply_dummy_coding(self.data, 'svr')
        
        val_ids = ["7320762352381137486", "1326601974741727223",
                "13795479952677497758", "17970471663419943956",
                "591996572399293829", "6954492747098322828",
                "4519400601461371159", "11454960459781727628",
                "9768395304785573367", "7906825960605741597",
                "3905643410280481454", "12454419868848728357",
                "13486961177026390118", "18187469172894655333",
                "15676465151754481270", "4488190483628909777",
                "5667355130474897285", "964258365948286848",
                "17818785705188559565", "13640541194109558825",
                "16881539705098457661", "10198992337604092255",
                "16133277327554207772", "17027936431095655851",
                "4006556787408523793", "17828380218855946271",
                "17795348434156827663", "9819797841580929529",
                "7622575789119914178", "10836067036626445968",
                "10840594361218801501", "6849908377245687254",
                "15832460680890497046", "9815281397449858316",
                "13009415889885870525", "17193588718582764562",
                "11644173636197049081", "17003039771047908208",
                "14351366332599145086", "8194206286320205104",
                "5692045396867215843", "16062413441531954747",
                "13065498101602502269", "15630495468350842421",
                "2400915835830124098", "11530282848970745391",
                "1115770551400009487", "4526062704021156700",
                "15956991559213678896", "3357274925576012669",
                "2033255983674970214", "12658661573620019110",
                "6255741892431163040", "8288152357428885260",
                "13340114743452940308", "8314969399756073668",
                "5238296034497477044", "3636511060856514098",
                "6364219505840161130", "3798501454412744418",
                "777415161209995588", "8667841844491088304",
                "229831732984625798", "13650586586314627826",
                "17891827491669968268", "2260755193876131907",
                "121393340552408620", "787711980568006406",
                "10042042071473919540", "10115557741499455579",
                "11540055609545135557", "8781566056153779038",
                "15514096673674662274", "3639847870770156967",
                "4230688520128215935", "4951696076046304705",
                "18259807734269833476", "6349231821864840119",
                "16960399523252868813", "11544889873844603832",
                "16296489504445822662", "9771940656370572981",
                "11197160959360408895", "16339705762773060111",
                "14887574878643139909", "2812302064763831962",
                "8023409695905298613", "3885712002505993349",
                "4492264831939922519", "1556139017108071009",
                "13077806425412745049", "15700956778227868322",
                "1269077004618136411", "13999952642140301421",
                "11075953045429478446", "13578758997251769242",
                "5864663906864527368", "4201105441247825184",
                "14667683723488806969", "10660778921539065951"]
        
        cross_datasets = []
        
        if cross_type == '2fold':
            dc_df_subsets = self.apply_2fold(dc_data_frame, val_ids)
            
            normalized_ds = self.normalize_dataset(dc_df_subsets[0],
                    dc_df_subsets[1], y_label)

            cross_datasets.append({'training.x': normalized_ds[1],
                                   'training.y': normalized_ds[0],
                                   'test.x': normalized_ds[3],
                                   'test.y': normalized_ds[2],
                                   'dataset': dc_data_frame})
            
        else:
            dc_df_subsets = self.apply_kfold(dc_data_frame, 10)

            for i in range(len(dc_df_subsets)):
                validation_set = dc_df_subsets[i]
                
                training_subsets = list(dc_df_subsets)
                del training_subsets[i]

                training_set = r['rbind'](training_subsets[0], training_subsets[1],
                                          training_subsets[2], training_subsets[3],
                                          training_subsets[4], training_subsets[5],
                                          training_subsets[6], training_subsets[7],
                                          training_subsets[8])

                normalized_ds = self.normalize_dataset(validation_set, 
                        training_seti, y_label)

                cross_datasets.append({'training.x': normalized_ds[1],
                                       'training.y': normalized_ds[0],
                                       'test.x': normalized_ds[3],
                                       'test.y': normalized_ds[2],
                                       'dataset': dc_data_frame})

        gen_res_file = open(svm_wd + 'general.results.txt', 'a')

        gamma_opt = [pow(2, i) for i in range(-15, 4) if i != 0]
        cost_opt  = [pow(2, i) for i in range(-5, 16) if i != 0]

        for svm_type in [3]:
            for kernel_type in [2]:
                for degree in [3]:
                    for gamma in [0.0625]:
                        for cost in [2048]:
                            dir_idx += 1
                            os.mkdir(svm_wd + str(dir_idx) + '/')
                            os.chdir(svm_wd + str(dir_idx) + '/')
                            os.mkdir(svm_wd + str(dir_idx) + '/' + 'plots/')
                            
                            c_cross_results = []

                            for ds_idx in range(len(cross_datasets)):
                                ds = cross_datasets[ds_idx]
                                # computing the model
                                prob = svm_problem(ds['training.y'],
                                                   ds['training.x'])
                                param = svm_parameter(\
                                        ' -s ' + str(svm_type) +\
                                        ' -t ' + str(kernel_type) +\
                                        ' -d ' + str(degree) +\
                                        ' -g ' + str(gamma) +\
                                        ' -c ' + str(cost))
                                m = svm_train(prob, param)

                                # computing the cross-validation results
                                p_label, p_acc, p_val =\
                                    svm_predict(ds['test.y'], ds['test.x'], m)

                                # computing the residual (error) measures
                                rel_errors = [error for error in\
                                        np.subtract(p_label, ds['test.y']) /\
                                        ds['test.y']]
                                abs_rel_errors = [error for error in\
                                        np.absolute(np.subtract(p_label,
                                            ds['test.y'])) / ds['test.y']]

                                mean_rel_error = np.mean( rel_errors )
                                mean_abs_rel_error = np.mean( abs_rel_errors )

                                c_cross_results.append({
                                    'rel_errors': rel_errors,
                                    'abs_rel_errors': abs_rel_errors,
                                    'mean_rel_error': mean_rel_error,
                                    'mean_abs_rel_error': mean_abs_rel_error})
                            
                            global_rel_errors = []
                            global_abs_rel_errors = []
                            for ds_idx in range(len(cross_datasets)):
                                global_rel_errors +=\
                                    c_cross_results[ds_idx]['rel_errors']
                                global_abs_rel_errors +=\
                                    c_cross_results[ds_idx]['abs_rel_errors']
                            
                            global_mean_rel_error = np.mean(global_rel_errors)
                            global_mean_abs_rel_error =\
                                np.mean(global_abs_rel_errors)
                            
                            # exporting laptops csv with prediction error results
                            r['write.csv'](ds['dataset'], file='ds.csv')
                            r['write.csv'](r['data.frame'](global_abs_rel_errors), file='errors.csv')
                            r['write.csv'](r['data.frame'](global_rel_errors), file='errors.csv')
                            
                            # making histograms of the residuals
                            hist, bins = np.histogram(global_rel_errors, bins = 50)
                            width = 0.7 * (bins[1] - bins[0])
                            center = (bins[:-1] + bins[1:])/2
                            plt.bar(center, hist, align = 'center', width = width)
                            plt.savefig('plots/rel_error_hist_plot.png')
                            plt.clf()

                            hist, bins = np.histogram(global_abs_rel_errors, 
                                    bins = 50)
                            width = 0.7 * (bins[1] - bins[0])
                            center = (bins[:-1] + bins[1:])/2
                            plt.bar(center, hist, align = 'center', width = width)
                            plt.savefig('plots/abs_rel_error_hist_plot.png')
                            plt.clf()

                            res_file = open(svm_wd + str(dir_idx) + '/' +\
                                    'results.txt', 'w')

                            res_file.write('svm_type: ' + str(svm_type) + ', ' +\
                                    'kernel_type: ' + str(kernel_type) + ', ' +\
                                    'degree: ' + str(degree) + ', ' +\
                                    'gamma: ' + str(gamma) + ', ' +\
                                    'cost: ' + str(cost) + '\r\n\r\n')

                            # printing the mean results
                            res_file.write('mean rel error: ' +\
                                    str(global_mean_rel_error) + '\r\n')
                            res_file.write('mean abs rel error: ' +\
                                    str(global_mean_abs_rel_error) + '\r\n')
                            res_file.write('accuracy: ' + str(p_acc[0]) + ', '\
                                    + str(p_acc[1]) + ', ' + str(p_acc[2]))
                            
                            if cross_type != '2fold':
                                for i in range(10):
                                    res_file.write('\r\n')
                                    res_file.write(str(c_cross_results[i]['mean_rel_error']))
                                    res_file.write(str(c_cross_results[i]['mean_abs_rel_error']))
                                    res_file.write('\r\n')

                            svm_params = '(' + ', '.join([str(svm_type), 
                                str(kernel_type),str(degree),str(gamma),
                                str(cost)]) + ')'

                            gen_res_file.write(str(dir_idx) + '\t' +\
                                    str(global_mean_abs_rel_error) + '\t' +\
                                    svm_params +  '\r\n')
                            
                            gen_res_file.flush()
                            os.fsync(gen_res_file.fileno())

                            res_file.close()
        gen_res_file.close()


