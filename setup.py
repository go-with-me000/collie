from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8") as f:
    reqs = f.read()

setup(
    name="collie",
    version="1.0.1",
    description="CoLLiE: Collaborative Tuning of Large Language Models in an Efficient Way",
    author="OpenLMLab",
    author_email="yanhang@pjlab.org.cn",
    packages=find_packages(),
    install_requires=reqs.splitlines(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "collie = collie.commands.collie_cli:main",
        ],
    },
)
