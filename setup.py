from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name="nlfepy",  # Required
    version="0.1.2",  # Required
    description="A simple finite element library for nonlinear analysis",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",
    url="https://github.com/sKujirai/nlfepy",  # Optional
    # author='hoge',  # Optional
    # author_email='author@example.com',  # Optional
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
    ],
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",  # 3: Alpha, 4: Beta, 5: Production/Stable
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="sample, setuptools, development",  # Optional
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.5",
)
