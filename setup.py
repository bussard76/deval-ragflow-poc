from setuptools import find_packages, setup

setup(
    name="deval-ragflow",
    version="0.3.1",
    description="Small local PDF provenance and RAGFlow proof of concept",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9,<3.13",
    install_requires=["httpx==0.28.1", "PyMuPDF==1.26.5"],
    extras_require={"test": ["pytest==8.4.2"]},
    entry_points={
        "console_scripts": [
            "deval-ragflow=deval_ragflow.cli:main",
            "deval-webchat=deval_ragflow.web:main",
        ]
    },
)
