import sympy as sp

import MLGeometry


class Environment:
    def __init__(self, config):
        self.config = config
        z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
        Z = [z0, z1, z2, z3, z4]
        if config.name == "fermat_quintett":
            f = z0 ** 5 + z1 ** 5 + z2 ** 5 + z3 ** 5 + z4 ** 5 + 0.5 * z0 * z1 * z2 * z3 * z4
        elif config.name == "phi_function":
            phi = 0.5
            psi = 0.5
            f = z0 ** 5 + z1 ** 5 + z2 ** 5 + z3 ** 5 + z4 ** 5 + psi * z0 * z1 * z2 * z3 * z4
            f = f + phi * (z3 * z4 ** 4 + z3 ** 2 * z4 ** 3 + z3 ** 3 * z4 ** 2 + z3 ** 4 * z4)
        elif config.name == "alpha_function":
            alpha = 0.5
            psi = 0.5
            f = z0 ** 5 + z1 ** 5 + z2 ** 5 + z3 ** 5 + z4 ** 5 + psi * z0 * z1 * z2 * z3 * z4
            f = f + alpha * (
                        z2 * z0 ** 4 + z0 * z4 * z1 ** 3 + z0 * z2 * z3 * z4 ** 2 + z3 ** 2 * z1 ** 3 + z4 * z1 ** 2 * z2 ** 2 + z0 * z1 * z2 * z3 ** 2 +
                        z2 * z4 * z3 ** 3 + z0 * z1 ** 4 + z0 * z4 ** 2 * z2 ** 2 + z4 ** 3 * z1 ** 2 + z0 * z2 * z3 ** 3 + z3 * z4 * z0 ** 3 + z1 ** 3 * z4 ** 2 +
                        z0 * z2 * z4 * z1 ** 2 + z1 ** 2 * z3 ** 3 + z1 * z4 ** 4 + z1 * z2 * z0 ** 3 + z2 ** 2 * z4 ** 3 + z4 * z2 ** 4 + z1 * z3 ** 4)
        else:
            raise NotImplementedError()

        HS_train = MLGeometry.hypersurface.Hypersurface(Z, f, config.n_pairs)
        HS_test = MLGeometry.hypersurface.Hypersurface(Z, f, config.n_pairs)
        self.train_set = MLGeometry.tf_dataset.generate_dataset(HS_train)
        self.test_set = MLGeometry.tf_dataset.generate_dataset(HS_test)
        self.train_set = self.train_set.shuffle(HS_train.n_points).batch(config.batch_size)
        self.test_set = self.test_set.shuffle(HS_test.n_points).batch(config.batch_size)

