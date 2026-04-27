load("@rules_python//python:defs.bzl", "py_test")
load("@pip//:requirements.bzl", "requirement")

def py_lint_test(name, srcs, config = "//:.pylintrc", **kwargs):
    py_test(
        name = name,
        srcs = ["//tools:pylint_wrapper.py"] + srcs,
        main = "//tools:pylint_wrapper.py",
        args = ["--rcfile=$(location %s)" % config] + ["$(location %s)" % s for s in srcs],
        data = [config] + srcs,
        deps = [requirement("pylint")],
        **kwargs
    )
