from rpy import r
import rpy

class Regression:
    "Regression"
    
    def __init__(self, data):
        self.data = data
        self.data_training_items = [0, round(len(data) * 0.8)]
        self.data_test_items = [round(len(data) * 0.8), len(data)]

    def run_linear_reg():
        rpy.set_default_mode(rpy.NO_CONVERSION)

        linear_model = r.lm(rpy.r("base.price ~ cpu.clock"), data =
                r.data_frame(x = self.data_training['cpu.clock'],
                    y = self.data_training['base.price']))

        print linear_model

    def run_poly_reg(degree):
        return 0 
