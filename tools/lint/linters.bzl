load("@aspect_rules_lint//lint:bandit.bzl", "lint_bandit_aspect")
load("@aspect_rules_lint//lint:buildifier.bzl", "lint_buildifier_aspect")
load("@aspect_rules_lint//lint:ruff.bzl", "lint_ruff_aspect")
load("@aspect_rules_lint//lint:ty.bzl", "lint_ty_aspect")

ruff = lint_ruff_aspect(
    binary = "@aspect_rules_lint//lint:ruff_bin",
    configs = [Label("//:ruff.toml")],
)

bandit = lint_bandit_aspect(
    binary = Label("//tools/lint:bandit"),
    config = Label("//:pyproject.toml"),
)

ty = lint_ty_aspect(
    binary = Label("@aspect_rules_lint//lint:ty_bin"),
    config = Label("//:pyproject.toml"),
)

buildifier = lint_buildifier_aspect(
    binary = Label("@buildifier_prebuilt//:buildifier"),
)
