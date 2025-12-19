from setuptools import find_packages, setup


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> list[str]:
        with open(filename, encoding="utf-8") as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    try:
        requirements = _read_requirements("requirements.txt")
        return requirements
    except ValueError:
        print("Failed to read requirements.txt in cfgan.")
        return []


setup(
    name="cfgan",
    version="0.1.0",
    description="CFGAN: Cross-Filter Generative Adversarial Network Framework for Microscopy Image Processing",
    packages=find_packages(),
    install_requires=get_requirements(),
    author="zhoux77899",
    author_email="zhouxiang100@huawei.com",
    url="https://github.com/zhoux77899/CFGAN",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
