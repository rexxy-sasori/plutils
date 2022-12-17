import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fr:
    requirements = fr.readlines()
    requirements = list(map(lambda x:x[:-1], requirements))

setuptools.setup(
    name="plutils",                     # This is the name of the package
    version="0.0.2",                        # The initial release version
    author="Rex Geng",                     # Full name of the author
    description="PytorchLightning API for Developing Neural Network",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    install_requires=requirements                    # Install other dependencies if any
)