from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

setup(
    name="statutil",
    version="0.0.1",
    description="a CLI package for handling FAERS data",
    author="tadahaya",
    packages=find_packages(),
    install_requires=install_requirements,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "statutil.tintegration=statutil.multivariate:ting",
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ]
)