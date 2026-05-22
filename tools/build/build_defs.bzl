load("@aspect_rules_py//py:defs.bzl", _py_library = "py_library", _py_test = "py_test")

def py_library(name, imports = None, **kwargs):
    if imports == None:
        imports = []
    _py_library(
        name = name,
        imports = imports + ["src"],
        **kwargs,
    )

def py_test(name, deps = None, imports = None, pytest_main = True, **kwargs):
    if deps == None:
        deps = []
    if imports == None:
        imports = []
    _py_test(
        name = name,
        deps = deps + ["@pypi//pytest"],
        imports = imports + ["."],
        pytest_main = pytest_main,
        **kwargs
    )