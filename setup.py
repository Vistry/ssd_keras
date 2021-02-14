import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="ssd_keras",
    version="0.0.1",
    author="Gabriel Ibagon",
    author_email="gabriel.ibagon@vistry.ai",
    description="Keras implementation of SSD for object detection",
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vistry/ssd_keras",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
)
