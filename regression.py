import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

r = robjects.r

class Regression:
    "Regression"
    
    def __init__(self, data):
        self.data = data
        
        self.stats = importr('stats')
        self.base = importr('base')

        self.data_training_items = [0, round(len(self.data) * 0.8)]
        self.data_test_items = [round(len(self.data) * 0.8), len(self.data)]
        
    def mount_reg_params(self, degree):
        kwargs = {}
        model_x = []

        base_price_pos = self.get_item_pos(r['names'](self.data), 'base.price')
        kwargs['base.price'] = self.data[base_price_pos]

        for i in range(degree):
            cont = 0
            for attr in r['names'](self.data):
                if attr == 'base.price':
                    continue
                else:
                    attr_deg = attr + '.' + str(i+1)
                
                if r['class'](self.data[cont])[0] != 'factor' or i == 0:
                    if r['class'](self.data[cont])[0] != 'factor':
                        model_x.append(attr_deg)
                        kwargs[attr_deg] = np.power(self.data[cont], (i+1)).tolist()
                    else:
                        contrasts = r['contrasts'](self.data[cont])
                        levels = r['levels'](self.data[cont])
                        c_data = r['as.vector'](self.data[cont]) 
                        
                        for j in range(1,len(levels)):
                            var_name = attr_deg + '.' + levels[j]
                            kwargs[var_name] = [None] * len(c_data)
                            model_x.append(var_name)

                        for j in range(0,len(c_data)):
                            cont_row = self.get_item_pos(levels, c_data[j]) + 1
                            for k in range(1,len(levels)):
                                kwargs[attr_deg + '.' +
                                    levels[k]][j] = contrasts.rx(cont_row, k)[0]

                cont += 1
        
        for attr in kwargs:
            if attr != 'base.price':
                kwargs[attr] = robjects.FloatVector(kwargs[attr])
        
        return r['data.frame'](**kwargs), ('base.price~' +
                '+'.join(model_x).replace(' ', '.'))
    
    def lm(self, l, h):
        for i in range(l,h+1):
            data_frame, data_model = self.mount_reg_params(i)
            
            linear_model = self.stats.lm(r['as.formula'](data_model), data = data_frame)
            print r['summary'](linear_model)[7]

    def get_item_pos(self, list, s):
        for idx, item in enumerate(list):
            if item == s:
               return idx
        return -1
