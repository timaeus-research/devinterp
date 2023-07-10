from setuptools import setup

setup(
    name="devinterp",
    version="0.0.0",
    packages=["devinterp"],
    # license="LICENSE",
    description="A library for doing research on developmental interpretability.",
    long_description=open("README.md").read(),
    install_requires=[
        "einops",
        "numpy",
        "torch",
        "datasets",
        "transformers",
        "tqdm",
        "pandas",
        "datasets",
        "wandb",
        "fancy_einsum",
        "rich",
        "accelerate",
        "typing-extensions",
    ],
    extras_require={"dev": ["pytest", "mypy", "pytest-cov"]},
)