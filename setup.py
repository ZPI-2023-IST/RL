import setuptools

setuptools.setup(
    name="rl-api",
    version="1.0.0",
    author="Wiktor and Patryk",
    description="RL module",
    packages=setuptools.find_packages(where="."),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    py_modules=["rl"],
    package_dir={'': '.'},

    install_requires=[
        "flask",
        "flask-cors",
        "numpy",
        "torch",
    ]
)
