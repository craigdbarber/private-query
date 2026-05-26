"""
This module provides convencience wrappers for python build targets.
"""

load("@aspect_rules_py//py:defs.bzl", _py_library = "py_library", _py_test = "py_test")

def py_library(name, imports = None, **kwargs):
    """Wrap py_library ensuring the "src" dir is included in imports.

    Args:
        name: The name of the target.
        imports: The impports for the target.
        **kwargs: Pass through keyword args.
    """
    if imports == None:
        imports = []
    _py_library(
        name = name,
        imports = imports + ["src"],
        **kwargs
    )

def py_test(name, deps = None, imports = None, pytest_main = True, **kwargs):
    """Wrap py_test ensuring "@pypi//pytest" is included as a dependency.

    Args:
        name: The name of the target.
        deps: The target's dependencies.
        imports: The target's imports.
        pytest_main: If the target is a pytest executable, defaults to True.
        **kwargs: Pass through keyword args.
    """
    if deps == None:
        deps = []
    if imports == None:
        imports = []
    _py_test(
        name = name,
        deps = deps + ["@pypi//pytest"],
        imports = imports + ["."],
        pytest_main = pytest_main,
        package_collisions = "ignore",  # compatibility with uv and linux sandbox engine
        **kwargs
    )
