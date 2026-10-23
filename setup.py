from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="textsteer",
    version="0.1.0",
    description="TextSteer: diagnosis-aware, reward-guided textual prompt optimization built on TextGrad.",
    python_requires=">=3.10",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    license="MIT license",
    packages=find_packages(include=["textgrad", "textgrad.*", "textsteer", "textsteer.*"]),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "vllm": ["vllm"],
    },
    zip_safe=False,
)
