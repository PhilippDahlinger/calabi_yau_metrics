from calabi_yau_metrics.algorithms.task_1_algorithm import Task1Algorithm
from calabi_yau_metrics.algorithms.task_2_algorithm import Task2Algorithm
from calabi_yau_metrics.algorithms.task_3_algorithm import Task3Algorithm


def get_algorithm(config, env):
    if config.name == "task_1":
        algorithm = Task1Algorithm(config, env)
    elif config.name == "task_2":
        algorithm = Task2Algorithm(config, env)
    elif config.name == "task_3":
        algorithm = Task3Algorithm(config, env)
    else:
        raise ValueError(f"Algorithm {config.name} not found")
    return algorithm