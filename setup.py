# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from setuptools import setup, find_packages

setup(
    name="kbc",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
    ],
)
