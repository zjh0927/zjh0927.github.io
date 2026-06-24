from setuptools import setup

package_name = 'vla_agent'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zeng Jiahao',
    maintainer_email='3083448680@qq.com',
    description='ROS2 + Gazebo + VLM: Brain-Cerebellum Embodied Intelligence',
    license='MIT',
    entry_points={
        'console_scripts': [
            'vla_agent_node2 = vla_agent.vla_agent_node2:main',
            'instruction_publisher = vla_agent.instruction_publisher:main',
        ],
    },
)
