load("@aspect_rules_py//py:defs.bzl", _py_library = "py_library", _py_test = "py_test")

def py_library(name, srcs = None, deps = None, imports = None, **kwargs):
    if imports == None:
        imports = []
    if deps == None:
        deps = []
    if srcs == None:
        srcs = []
    _py_library(
        name = name,
        srcs = srcs,
        deps = deps,
        imports = imports + ["src"],
        **kwargs
    )

def py_test(name, srcs = None, deps = None, imports = None, pytest_main=True, **kwargs):
    if deps == None:
        deps = []
    if imports == None:
        imports = []
    if srcs == None:
        srcs = []
    _py_test(
        name = name,
        srcs = srcs,
        deps = deps + ["@pypi//pytest"],
        imports = imports + ["."],
        pytest_main = pytest_main,
        **kwargs
    )