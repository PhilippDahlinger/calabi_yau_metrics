from calabi_yau_metrics.algorithms.task_1_algorithm import Task1Algorithm


def get_algorithm(config, env):
    if config.name == "task_1":
        algorithm = Task1Algorithm(config, env)
    else:
        raise ValueError(f"Algorithm {config.name} not found")
    return algorithm