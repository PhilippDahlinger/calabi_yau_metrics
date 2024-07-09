import tensorflow as tf

import MLGeometry


class Evaluator:
    def __init__(self, config, env, algorithm):
        self.config = config
        self.env = env
        self.algorithm = algorithm

    def evaluate(self, epoch):
        if epoch % self.config.eval_interval != 0:
            return None, None
        mape_test_loss = self.algorithm.calc_total_loss(self.env.test_set, MLGeometry.loss.weighted_MAPE)
        mse_test_loss = self.algorithm.calc_total_loss(self.env.test_set, MLGeometry.loss.weighted_MSE)
        # TODO: rest evaluation of the Guide
        return mape_test_loss, mse_test_loss
