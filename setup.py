from setuptools import setup, find_packages

setup(
    name='rllib',
    packages=find_packages(),
    version='0.0.1',
    author='Zhang Dongkun',
    author_email='zhangdongkun98@gmail.com',
    url='https://github.com/zhangdongkun98/rl-lib',
    description='A lib for RL.',
    install_requires=[
        'numpy', 'scipy',
        'matplotlib', 'seaborn',
        'tensorboardX',
        'PyYAML',
        'psutil', 'pynvml',
        'memory-profiler',
    ],

    include_package_data=True
)
