load("@rules_python//python:defs.bzl", "py_test")

def pytest_test(name, srcs, deps = [], **kwargs):
    py_test(
        name = name,
        srcs = ["//tools:pytest_wrapper.py"] + srcs,
        main = "//tools:pytest_wrapper.py",
        # Pass the original test files as arguments so pytest discovers them
        args = ["$(location %s)" % s for s in srcs],
        deps = deps + ["@pip//pytest"], # Ensure pytest is in dependencies
        **kwargs,
    )