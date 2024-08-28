from setuptools import setup

setup(
    name="lm-benchmark",
    version="0.1.1",
    py_modules=["lm_benchmark"],
    install_requires=[
        "aiohttp",
        "numpy",
        "matplotlib",
        "pandas",
        "openpyxl",
    ],
    entry_points={
        "console_scripts": [
            "lm-benchmark=lm_benchmark:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["databricks-dolly-15k.jsonl"],
    },
)