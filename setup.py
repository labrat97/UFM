"""Package installation setup."""

from setuptools import setup

setup(
    name="uniflowmatch",
    version="0.0.0",
    description="UniFlowMatch Project",
    author="AirLab",
    license="BSD Clause-3",
    packages=["uniception", "uniflowmatch"],  # Directly specify the package
    package_dir={
        "uniception": "UniCeption/uniception",  # Map uniception package
        "uniflowmatch": "uniflowmatch",  # Map uniflowmatch package
    },
    include_package_data=True,
)
