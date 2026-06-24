from setuptools import setup

package_name = "vla_bridge"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "requests"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="ROS2 bridge: camera+instruction -> VLA /act -> cmd_vel",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vla_agent = vla_bridge.vla_agent_node:main",
            "vla_agent2 = vla_bridge.vla_agent_node2:main",
            "send_instruction = vla_bridge.instruction_publisher:main",
        ],
    },
)
