import os
import sys
import tensorflow as tf
import traceback

import hydra
from omegaconf import OmegaConf, DictConfig

from calabi_yau_metrics.algorithms import get_algorithm
from calabi_yau_metrics.envs.environment import Environment
from calabi_yau_metrics.util.util import load_omega_conf_resolvers

# full stack trace
os.environ["HYDRA_FULL_ERROR"] = "1"

# register OmegaConf resolver for hydra
load_omega_conf_resolvers()


@hydra.main(version_base=None, config_path="configs", config_name="training_config")
def train(config: DictConfig) -> None:
    try:
        print(OmegaConf.to_yaml(config))
        env = Environment(config.env)
        algorithm = get_algorithm(config.algorithm, env)
        for epoch in range(config.epochs):
            for step, (points, Omega_Omegabar, mass, restriction) in enumerate(env.train_set):

            if epoch % 50 == 0:
                print("epoch %d: loss = %.5f" % (epoch, loss))
        # for epoch in range(config.epochs):
        #     training_metrics = algorithm.train_step(epoch=epoch)
        #     evaluation_metrics = evaluator.eval_step(epoch=epoch)
        #     # combine training and evaluation metrics
        #     metrics = deep_update(training_metrics, evaluation_metrics)
        #     # start a new thread to record the iteration, this speeds up the overall training
        #     recorder.record_iteration(iteration=epoch, recorded_values=metrics)
        #
        # # # close wandb, save the final model, ...
        # recorder.finalize()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    train()
