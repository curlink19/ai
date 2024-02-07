import os


def get_global_variable(name: str, dtype=int):
    assert name in os.environ, "(env) no such variable: " + name
    return dtype(os.environ.get(name))