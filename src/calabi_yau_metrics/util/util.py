def load_omega_conf_resolvers():
    OmegaConf.register_new_resolver("sub_dir_shortener", shortener)
    OmegaConf.register_new_resolver(
        "format", lambda inpt, formatter: formatter.format(inpt)
    )
    OmegaConf.register_new_resolver("conditional_resolver", conditional_resolver)
