from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='voz_hmm_pkg',
            executable='grabador_node',
            name='grabador_node',
            output='screen'
        ),
        Node(
            package='voz_hmm_pkg',
            executable='entrenar_hmm_node',
            name='entrenar_hmm_node',
            output='screen'
        ),
        Node(
            package='voz_hmm_pkg',
            executable='mapa_comandos_node',
            name='mapa_comandos_node',
            output='screen'
        )
    ])

