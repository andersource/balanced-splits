import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="balanced-splits",
    version="0.1.0",
    author="andersource",
    author_email="hi@andersource.dev",
    description="Balanced splitting utility",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andersource/balanced-splits",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
