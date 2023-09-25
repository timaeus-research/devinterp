from setuptools import setup


def get_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name="devinterp",
    version="0.0.0",
    packages=["devinterp"],
    # license="LICENSE",
    description="A library for doing research on developmental interpretability.",
    long_description=open("README.md").read(),
    install_requires=get_requirements(),
    extras_require={"dev": ["pytest", "mypy", "pytest-cov", "moto[ec2,s3,all]"]},
)
