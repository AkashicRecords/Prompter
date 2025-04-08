from setuptools import setup, find_packages

setup(
    name="prompter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.0.0",
        "rich>=10.0.0",
        "typer>=0.9.0",
        "pydantic>=2.0.0",
        "websockets>=10.0",
        "cryptography>=40.0.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "prompter=prompter.cli:main",
        ],
    },
    author="AkashicRecords",
    author_email="info@akashicrecords.com",
    description="An interactive CLI for prompt engineering with AI learning capabilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AkashicRecords/Prompter",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 