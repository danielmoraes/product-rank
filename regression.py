import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from utils import *
from svm import *
from svmutil import *
import matplotlib.pyplot as plt
import Image

r = robjects.r

class Regression:
    "Regression"
    
    def __init__(self, data, wdir):
        self.data = data
        
        # rpy2 imports
        self.stats = importr('stats')
        self.base = importr('base')
        self.e1071 = importr('e1071')
        
        # setting the results working directory
        self.wd = wdir
        self.wd += 'results/'
        if not os.path.exists(self.wd):
            os.makedirs(self.wd)
        os.chdir(self.wd)

    def make_regression_data_frame(self, degree):
        kwargs = {}
        model_x = []
        
        base_price_pos = get_item_pos(r['names'](self.data), 'base.price')
        
        kwargs['base.price'] = self.data[base_price_pos]

        for i in range(degree):
            cont = 0
            for attr in r['names'](self.data):
                if attr == 'base.price':
                    continue
                else:
                    attr_deg = attr + '.' + str(i+1)
                
                data_attr = self.data[cont]

                if r['class'](data_attr)[0] != 'factor' or i == 0:
                    if r['class'](data_attr)[0] != 'factor':
                        model_x.append(attr_deg)
                        kwargs[attr_deg] = np.power(data_attr, (i+1)).tolist()
                    else:
                        contrasts = r['contrasts'](data_attr)
                        levels = r['levels'](data_attr)
                        data_v = r['as.vector'](data_attr)

                        for j in range(1,len(levels)):
                            var_name = attr_deg + '.' + levels[j]
                            kwargs[var_name] = [None] * len(data_v)
                            model_x.append(var_name)

                        for j in range(0,len(data_v)):
                            cont_row = get_item_pos(levels, data_v[j]) + 1
                            for k in range(1,len(levels)):
                                kwargs[attr_deg + '.' +
                                    levels[k]][j] = contrasts.rx(cont_row, k)[0]

                cont += 1
        
        for attr in kwargs:
            if attr != 'base.price':
                kwargs[attr] = robjects.FloatVector(kwargs[attr])
        
        data_model = ('base.price~' + '+'.join(model_x).replace(' ', '.'))
        data_frame = r['data.frame'](**kwargs)
        
        data_frame_training = data_frame.rx(r['seq'](0, int(round(len(self.data[0]) * 0.8))), True)
        data_frame_test = data_frame.rx(r['seq'](int(round(len(self.data[0]) * 0.8)) + 1, len(self.data[0])), True)

        return data_frame_training, data_frame_test, data_model
    
    def make_svr_data_frame(self):
        kwargs = {}
        
        base_price_pos = get_item_pos(r['names'](self.data), 'base.price')
        
        kwargs['base.price'] = self.data[base_price_pos]
        
        cont = 0
        for attr in r['names'](self.data):
            if attr == 'base.price':
                continue
            
            data_attr = self.data[cont]

            if r['class'](data_attr)[0] != 'factor':
                kwargs[attr] = list(data_attr)
            else:
                levels = r['levels'](data_attr)
                data_v = r['as.vector'](data_attr)
                
                for j in range(0,len(levels)):
                    var_name = attr + '.' + levels[j]
                    kwargs[var_name] = [None] * len(data_v)

                for j in range(0,len(data_v)):
                    pos = get_item_pos(levels, data_v[j])
                    for k in range(0,len(levels)):
                        if pos == k: val = 1
                        else: val = 0
                        kwargs[attr + '.' + levels[k]][j] = val

            cont += 1
        
        for attr in kwargs.keys():
            if attr != 'base.price':
                kwargs[attr] = robjects.FloatVector(kwargs[attr])
        
        data_frame = r['data.frame'](**kwargs)
        
        data_frame_training = data_frame.rx(r['seq'](0, int(round(len(self.data[0]) * 0.8))), True)
        data_frame_test = data_frame.rx(r['seq'](int(round(len(self.data[0]) * 0.8)) + 1, len(self.data[0])), True)

        return data_frame_training, data_frame_test

    def lm(self, degree):
        data_frame_training, data_frame_test, data_model = self.make_regression_data_frame(degree)
        fit = self.stats.lm(r['as.formula'](data_model), data = data_frame_training)
        
        pred_prices = r['predict'](fit, data_frame_test, type="response")
        real_prices = data_frame_test[get_item_pos(data_frame_test.names, 'base.price')]
        
        return fit, {'pred_prices': pred_prices, 'real_prices': real_prices}

    def svr(self):
        svm_wd = self.wd + 'svr/'
        if not os.path.exists(svm_wd):
            os.makedirs(svm_wd)
        os.chdir(svm_wd)
        
        dir_files = [int(item.split('.')[0]) for item in os.listdir(svm_wd) if
                item.split('.')[0] != 'general']
        if len(dir_files) == 0:
            dir_idx = 0
        else:
            dir_idx = max(dir_files)

        # applying the coding scheme to the data
        data_frame_training, data_frame_test = self.make_svr_data_frame()
        
        # mounting the data in the libsvm format
        base_price_pos = get_item_pos(data_frame_training.names, 'base.price')
        
        y_data_training = list(data_frame_training[base_price_pos])
        x_data_training = list(data_frame_training)
        x_data_training = x_data_training[0:base_price_pos] + x_data_training[base_price_pos+1:len(x_data_training)]
        x_data_training = [list(item) for item in x_data_training]
        x_data_training_t = [list(item) for item in np.transpose(x_data_training)]
        
        y_data_test = list(data_frame_test[base_price_pos])
        x_data_test = list(data_frame_test)
        x_data_test = x_data_test[0:base_price_pos] + x_data_test[base_price_pos+1:len(x_data_test)]
        x_data_test = [list(item) for item in x_data_test]
        x_data_test_t = [list(item) for item in np.transpose(x_data_test)]
        
        # scaling the data
        scaled_x_data_training_t, x_ranges_dict = normalize_data(x_data_training_t)
        scaled_x_data_test_t = normalize_data(x_data_test_t, x_ranges_dict)
        
        scaled_y_data_training, y_ranges_dict = normalize_data(y_data_training)
        scaled_y_data_test = normalize_data(y_data_test, y_ranges_dict)
        
        gen_res_file = open(svm_wd + 'general.results.txt', 'a')

        for svm_type in [3]:
            for kernel_type in [0,1,2]:
                for degree in [3]:
                    for gamma in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                        for cost in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                            dir_idx += 1
                            os.mkdir(svm_wd + str(dir_idx) + '/')
                            os.chdir(svm_wd + str(dir_idx) + '/')
                            os.mkdir(svm_wd + str(dir_idx) + '/' + 'plots/')

                            # computing the model
                            prob = svm_problem(y_data_training, scaled_x_data_training_t)
                            param = svm_parameter(
                                    ' -s ' + str(svm_type) + 
                                    ' -t ' + str(kernel_type) + 
                                    ' -d ' + str(degree) + 
                                    ' -g ' + str(gamma) + 
                                    ' -c ' + str(cost))
                            m = svm_train(prob, param)
                            
                            # computing the cross-validation results
                            p_label, p_acc, p_val = svm_predict(y_data_test, scaled_x_data_test_t, m)
                            
                            # computing the residual (error) measures
                            relative_errors = [error for error in np.subtract(p_label, y_data_test) / y_data_test]
                            abs_relative_errors = [error for error in np.absolute( np.subtract(p_label, y_data_test)  ) / y_data_test]

                            mean_relative_error = np.mean( relative_errors )
                            mean_abs_relative_error = np.mean( abs_relative_errors )
                            
                            # making histograms of the residuals
                            hist, bins = np.histogram(relative_errors, bins = 50)
                            width = 0.7 * (bins[1] - bins[0])
                            center = (bins[:-1] + bins[1:])/2
                            plt.bar(center, hist, align = 'center', width = width)
                            plt.savefig('plots/rel_error_hist_plot.png')
                            plt.clf()
                            
                            hist, bins = np.histogram(abs_relative_errors, bins = 50)
                            width = 0.7 * (bins[1] - bins[0])
                            center = (bins[:-1] + bins[1:])/2
                            plt.bar(center, hist, align = 'center', width = width)
                            plt.savefig('plots/abs_rel_error_hist_plot.png')
                            plt.clf()

                            res_file = open(svm_wd + str(dir_idx) + '/' + 'results.txt', 'w')
                            
                            res_file.write('svm_type: ' + str(svm_type) + ', ' +
                                           'kernel_type: ' + str(kernel_type) + ', ' +
                                           'degree: ' + str(degree) + ', ' +
                                           'gamma: ' + str(gamma) + ', ' + 
                                           'cost: ' + str(cost) + '\r\n\r\n')

                            # printing the mean results
                            res_file.write('mean rel error: ' + str(mean_relative_error) + '\r\n')
                            res_file.write('mean abs rel error: ' + str(mean_abs_relative_error) + '\r\n')
                            res_file.write('accuracy: ' + str(p_acc[0]) + ', '
                                    + str(p_acc[1]) + ', ' + str(p_acc[2]))
                            
                            svm_params = '(' + ', '.join([str(svm_type),str(kernel_type),str(degree),str(gamma),str(cost)]) + ')'

                            gen_res_file.write(str(dir_idx) + '\t' +
                                    str(mean_abs_relative_error) + '\t' +
                                    svm_params +  '\r\n')
        res_file.close()
        gen_res_file.close()
