from setuptools import setup, find_packages

setup(
    name='GlobalClassifier',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'GlobalClassifier': ['model_params.json'],
    },
    install_requires=[
        'joblib==1.3.2',
        'matplotlib==3.5.3',
        'numpy==1.21.6',
        'pandas==1.3.5',
        'scikit-learn==1.0.2',
        'scipy==1.7.3',
        'psutil==5.9.0',
        'pyyaml==6.0'
    ],
    description='Uma biblioteca para avaliação de bases de dados hierárquicas',
)