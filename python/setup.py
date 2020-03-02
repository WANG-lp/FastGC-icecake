from setuptools import setup, find_packages
import sys

setup(
    name="pyicecake",
    version="0.1.0",
    author="WANG Lipeng",
    author_email="wang.lp@outlook.com",
    description="Python SDK of Fast GPU cache(icecake)",
    license="MIT",
    install_requires=[
    ],
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
)
