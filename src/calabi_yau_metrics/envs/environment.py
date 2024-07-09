import sympy as sp

import MLGeometry


class Environment:
    def __init__(self, config):
        self.config = config
        if config.name == "fermat_quintett":
            z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
            Z = [z0, z1, z2, z3, z4]
            f = z0 ** 5 + z1 ** 5 + z2 ** 5 + z3 ** 5 + z4 ** 5 + 0.5 * z0 * z1 * z2 * z3 * z4
        else:
            raise NotImplementedError()

        HS_train = MLGeometry.hypersurface.Hypersurface(Z, f, config.n_pairs)
        HS_test = MLGeometry.hypersurface.Hypersurface(Z, f, config.n_pairs)
        self.train_set = MLGeometry.tf_dataset.generate_dataset(HS_train)
        self.test_set = MLGeometry.tf_dataset.generate_dataset(HS_test)

