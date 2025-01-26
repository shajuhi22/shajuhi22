from setuptools import setup

setup(
    name='kbc',
    version='0.1.0',
    description='Your package description',
    author='Your Name',
    author_email='your.email@example.com',
    packages=['kbc'],
    package_data={'kbc': ['data/**/*']},
    install_requires=[
        # List your package dependencies here
    ],
)
