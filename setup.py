try:
    import tomllib
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib

from setuptools import find_packages, setup
from setuptools.command.install import install

with open("Cargo.toml", "rb") as f:
    __version__ = tomllib.load(f)["package"]["version"]

with open("README.md", "r") as f:
    long_description = f.read()


class CustomInstall(install):
    def run(self):
        install.run(self)


setup(
    name="rlgym-learn",
    packages=find_packages(),
    version=__version__,
    description="A multi-processed learning framework for use with RLGym.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jonathan Keegan",
    url="https://github.com/JPK314/rlgym-learn",
    install_requires=[
        "pydantic>=2.8.2",
        "numpy>1.21",
        "torch>1.13",
        "typing_extensions>4.6",
        "wandb>0.15",
    ],
    python_requires=">=3.8,<3.13",
    cmdclass={"install": CustomInstall},
    license="Apache 2.0",
    license_file="LICENSE",
    keywords=[
        "rocket-league",
        "gym",
        "reinforcement-learning",
        "simulation",
        "rlgym",
        "rocketsim",
    ],
)
