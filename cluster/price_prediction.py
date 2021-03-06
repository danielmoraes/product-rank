# to make plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# python to R interface
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# numpy lib
import numpy as np
r = robjects.r

# python to libsvm interface
from lib.libsvm2_svm import *
from lib.libsvm2_svmutil import *

# utils
from utils import *

# to export images
import Image

# python libs
import math
import os

# pyevolve lib
from pyevolve import *

# to export csv
import csv

class PricePrediction:
    "PricePrediction"

    def __init__(self, data, wdir, val_ids):
        """ Class constructor """
        # stores the dataset and the dataset ids
        self.data, self.data_ids = self.catch_dataset_ids(data, 
                'cid.product')
        
        self.data_val_ids = val_ids

        # stores the y label that will be predicted
        self.y_label = 'base.price'

        # rpy2 imports
        self.stats = importr('stats')
        self.base = importr('base')
        self.e1071 = importr('e1071')

        # setting the directory that the result files will be written
        self.wd = wdir + 'results/'
        if not os.path.exists(self.wd): os.makedirs(self.wd)
        os.chdir(self.wd)

    def catch_dataset_ids(self, data_frame, id_label):
        id_pos = get_item_pos(r['names'](data_frame), id_label)
        ids_idx = [item.replace('"', '') for item in
                list(r['as.vector'](data_frame[id_pos]))]
        
        keep = [item for item in list(r['names'](data_frame)) 
                if item != id_label]
        keep = robjects.StrVector(keep)
        data_frame_x = data_frame.rx(True, keep)

        return [data_frame_x, ids_idx]

    def apply_dummy_coding(self, data_frame, y_label, dc_type, degree = 1):
        """ Return a dummy coded R data frame """

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
        """ Return an array with the val/train dataframe sets """
        
        train_pos = [0 if cid in val_ids else 1 for cid in self.data_ids]

        training_rows = [i+1 for i in range(len(self.data_ids))
                if train_pos[i] == 1]
        validation_rows = [i+1 for i in range(len(self.data_ids))
                if train_pos[i] == 0]
        
        training_set = data_frame.rx(robjects.IntVector(training_rows), True)
        validation_set = data_frame.rx(robjects.IntVector(validation_rows), True)

        return [validation_set, training_set]

    def apply_kfold(self, data_frame, k):
        """ Return an array with the k dataframe subsets"""
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
        """ Return the norm val x, val y, norm train x, train y datasets"""
        # mounting the data in the libsvm format
        base_price_pos = get_item_pos(training_set.names, self.y_label)

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
        """ Apply the genetic algorithm with the dummy coded self.data """
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
        """ Genetic algorithm evaluation function """
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
        """ Apply the poly regression to the dummy coded self.data with degree """
        
        dc_data_frame = self.apply_dummy_coding(self.data, self.y_label, 'treatment', degree)
        
        x_labels = list(r['names'](dc_data_frame))
        del x_labels[get_item_pos(x_labels, self.y_label)]
        dc_data_model = (self.y_label + '~' + '+'.join(x_labels).replace(' ', '.'))
        
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
                self.y_label)]
            
            cross_results.append({'pred_prices': list(pred_prices), 
                'real_prices': list(real_prices)})
        
        return fit, cross_results

    def svr(self, resDirName, crossType):
        """ Apply the support vector regression to the dummy coded self.data """
        
        # setting the directory that the results will be written
        svm_wd = self.wd + resDirName + '/'
        if not os.path.exists(svm_wd):
           os.makedirs(svm_wd)
        os.chdir(svm_wd)

        # setting the new dir_idx to be the larger dir_idx+1
        dir_files = [int(item.split('.')[0]) for item in os.listdir(svm_wd) 
                     if item.split('.')[0] != 'general']
        if len(dir_files) == 0: dir_idx = 0
        else: dir_idx = max(dir_files)
        
        # applying dummy coding scheme to the dataset
        dc_data_frame = self.apply_dummy_coding(self.data, self.y_label, 'svr')
        
        cross_datasets = []
        
        # cross and normalize the dataset
        if crossType == '2fold':
            dc_df_subsets = self.apply_2fold(dc_data_frame, self.data_val_ids)
            
            normalized_ds = self.normalize_dataset(dc_df_subsets[0],
                    dc_df_subsets[1], self.y_label)

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
                        training_seti, self.y_label)

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
                                    'real_prices': ds['test.y'],
                                    'pred_prices': p_label,
                                    'rel_errors': rel_errors,
                                    'abs_rel_errors': abs_rel_errors,
                                    'mean_rel_error': mean_rel_error,
                                    'mean_abs_rel_error': mean_abs_rel_error})
                            
                            global_real_prices = []
                            global_pred_prices = []
                            global_rel_errors = []
                            global_abs_rel_errors = []
                            for ds_idx in range(len(cross_datasets)):
                                global_real_prices +=\
                                    c_cross_results[ds_idx]['real_prices']
                                global_pred_prices +=\
                                    c_cross_results[ds_idx]['pred_prices']
                                global_rel_errors +=\
                                    c_cross_results[ds_idx]['rel_errors']
                                global_abs_rel_errors +=\
                                    c_cross_results[ds_idx]['abs_rel_errors']
                            
                            global_mean_rel_error = np.mean(global_rel_errors)
                            global_mean_abs_rel_error =\
                                np.mean(global_abs_rel_errors)
                        
                            detail_res_file = csv.writer(open("detail_res_file.csv", "wb"))
                            
                            detail_res_file.writerow(["cid-product",\
                                "real-price", "predicted-price",\
                                "abs-rel-error", "rel-error"])
                           
                            for i in range(len(global_real_prices)):
                                detail_res_file.writerow([self.data_val_ids[i], 
                                    global_real_prices[i], int(global_pred_prices[i]),
                                    global_abs_rel_errors[i], global_rel_errors[i]])

                            r['write.csv'](ds['dataset'], file='ds.csv')
                            
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
                                    + str(p_acc[1]) + ', ' + str(p_acc[2]) +
                                    '\r\n')
                            
                            if crossType != '2fold':
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

