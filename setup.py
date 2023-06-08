from setuptools import find_packages, setup

LICENSE = "Closed Source"
COPYRIGHT = "Copyright (c) 2023 MICC"

PACKAGES = find_packages(where="src")
entry_points = {
    'console_scripts': [
        'cl2r=cl2r.main:main',
    ],
}

setup(
    name='cl2r',
    version='1.0',
    description='CL^2R: Comaptible Lifelong Learning Representations',
    license=LICENSE,
    py_modules=["cl2r"],
    packages=PACKAGES,
    package_dir={'': 'src'},
    entry_points=entry_points,
    python_requires=">=3"
)