from setuptools import setup, find_packages

setup(
    name="impartial",
    version="0.1.0",
    author="Gunjan Shrivastava",
    description="Impartial - Interactive method for instance segmentations using partial annotation ",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="impartial"),
    # packages=find_packages(),
    package_dir={"": "impartial"},
    # packages=['impartial'],

    url='https://github.com/nadeemlab/ImPartial',
    keywords=['Impartial', 'Multiplex', 'Segmentation', 'Interactive', 'Pathology'],
    install_requires=[
        # List dependencies here, e.g., "numpy>=1.21.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
