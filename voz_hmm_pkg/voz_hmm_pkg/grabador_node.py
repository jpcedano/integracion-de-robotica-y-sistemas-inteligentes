# grabador_node.py
import os
import rclpy
from rclpy.node import Node
import sounddevice as sd
from scipy.io.wavfile import write

class GrabadorNode(Node):
    def __init__(self):
        super().__init__('grabador_node')
        self.comandos = ["adelante", "atras", "izquierda", "derecha", "pausa", "continuar", "salir"]
        self.muestras_por_comando = 30
        self.duracion = 1.5  # segundos
        self.frecuencia = 16000
        self.directorio_base = os.path.expanduser("~/datos_hmm")
        self.regrabar_faltantes()

    def regrabar_faltantes(self):
        for comando in self.comandos:
            carpeta = os.path.join(self.directorio_base, comando)
            os.makedirs(carpeta, exist_ok=True)
            existentes = {f for f in os.listdir(carpeta) if f.endswith(".wav")}
            faltantes = [f"{comando}_{i:02d}.wav" for i in range(1, self.muestras_por_comando + 1) if f"{comando}_{i:02d}.wav" not in existentes]
            if faltantes:
                self.get_logger().info(f"Faltan {len(faltantes)} grabaciones para '{comando}'")
                input("Presiona ENTER para comenzar a grabar...")
                for nombre in faltantes:
                    print(f"Grabando {nombre}... Di: {comando}")
                    audio = sd.rec(int(self.duracion * self.frecuencia), samplerate=self.frecuencia, channels=1, dtype='int16')
                    sd.wait()
                    write(os.path.join(carpeta, nombre), self.frecuencia, audio)
                    print(f"Guardado: {nombre}")

        self.get_logger().info("Grabaciones completas. Puedes proceder al entrenamiento.")


def main(args=None):
    rclpy.init(args=args)
    node = GrabadorNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

