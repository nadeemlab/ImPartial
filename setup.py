from setuptools import setup, find_packages

setup(
    name="impartial",
    version="0.1.0",
    author="Gunjan Shrivastava",
    description="Impartial - Interactive method for instance segmentations using partial annotation ",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="impartial"),
    package_dir={"": "impartial"},

    url='https://github.com/nadeemlab/ImPartial',
    keywords=['Impartial', 'Multiplex', 'Segmentation', 'Interactive', 'Pathology'],
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
