from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="debugrl",
    version="0.1.0",
    author="Cameron Redovian",
    author_email="naivoder@gmail.com",
    description="Minimalistic Gymnasium environments for debugging RL algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/naivoder/debugrl",
    py_modules=["debugrl"],
    install_requires=[
        "gymnasium>=0.27.0",
        "numpy>=1.18.0",
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Framework :: Gymnasium",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
