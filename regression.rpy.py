from rpy import r
import rpy
import numpy as np
#import rpy2.robjects

#r = rpy2.robjects.r

class Regression:
    "Regression"
    
    def __init__(self, data):
        r.options(stringsAsFactors = True)

        self.data = data
        r.attach(self.data)

        self.data_training_items = [0, round(len(data) * 0.8)]
        self.data_test_items = [round(len(data) * 0.8), len(data)]
        
    def mount_reg_params(self, degree):
        kwargs = {}
        model_x = []

        for i in range(degree):
            for attr in r.names(self.data):
                if attr == 'base.price':
                    if i > 0: continue
                    attr_deg = attr
                else:
                    attr_deg = attr + '.' + str(i+1)
                
                if isinstance(self.data[attr][0], (int, long, float)) or i == 0:
                    if attr != 'base.price':
                        model_x.append(attr_deg)
                    if isinstance(self.data[attr][0], (int, long, float)):
                        kwargs[attr_deg] = np.power(r[attr], (i+1)).tolist()
                    else:
                        kwargs[attr_deg] = r[attr]

        return r.data_frame(**kwargs), ('base.price~' + '+'.join(model_x))
    
    def lm(self, l, h):
        for i in range(l,h+1):
            data_frame, data_model = self.mount_reg_params(i)
            print data_model
            rpy.set_default_mode(rpy.NO_CONVERSION)
            linear_model = r.lm(r(data_model), data = data_frame)
	    rpy.set_default_mode(rpy.BASIC_CONVERSION)	
            print r.summary(linear_model)['r.squared']

    def get_item_pos(self, list, s):
        for idx, item in enumerate(list):
            if item == s:
               return idx
        return -1
