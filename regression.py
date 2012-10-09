import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from utils import *
from svm import *
from svmutil import *

r = robjects.r

class Regression:
    "Regression"
    
    def __init__(self, data):
        self.data = data
        
        self.stats = importr('stats')
        self.base = importr('base')
        self.e1071 = importr('e1071')

    def make_treatment_data_frame(self, degree):
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
                        data_attr = data_attr
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
    
    def lm(self, degree):
        data_frame_training, data_frame_test, data_model = self.make_treatment_data_frame(degree)
        fit = self.stats.lm(r['as.formula'](data_model), data = data_frame_training)
        
        pred_prices = r['predict'](fit, data_frame_test, type="response")
        real_prices = data_frame_test[get_item_pos(data_frame_test.names, 'base.price')]
        
        return fit, {'pred_prices': pred_prices, 'real_prices': real_prices}

    def svr(self):
        data_frame_training, data_frame_test, data_model = self.make_treatment_data_frame(1)
        
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

        prob = svm_problem(y_data_training, x_data_training_t)
        param = svm_parameter()
        m = svm_train(prob, param)
        
        p_label, p_acc, p_val = svm_predict(y_data_test, x_data_test_t, m)
        
        error = np.mean(np.absolute( np.subtract(p_label, y_data_test)  ) / y_data_test)

        '''
        kwargs = {'formula': data_model, 'data': data_frame_training,
                  'scale': True, 'type': 'eps-regression', 'kernel': 'linear',
                  'degree': 3, 'gamma': 1, 'cost': 1, 'nu': 0.5}
        fit = self.e1071.svm(**kwargs)
        '''

        return 0
