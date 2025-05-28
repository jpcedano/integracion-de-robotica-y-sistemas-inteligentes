# mapa_comandos_node.py
import os
import numpy as np
import joblib
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import sounddevice as sd
from scipy.io.wavfile import write
from python_speech_features import mfcc
from scipy.signal import lfilter, butter, filtfilt

class MapaComandosNode(Node):
    def __init__(self):
        super().__init__('mapa_comandos_node')
        self.modelos_path = os.path.expanduser('~/modelos_hmm_gaussian')
        self.frecuencia = 16000
        self.duracion = 1.5
        self.min_frames = 20
        self.nombres_modelos = {}
        self.modelos = {}
        self.pub = self.create_publisher(Int32, 'comando_numerico', 10)

        # Cargar modelos más recientes
        self.cargar_modelos()
        self.get_logger().info('Modelos HMM cargados. Iniciando escucha en tiempo real...')
        self.timer = self.create_timer(5.0, self.escuchar_audio)

    def preenfasis(self, signal, coef=0.97):
        return lfilter([1, -coef], 1, signal)

    def filtro_pasabanda(self, signal, rate, low=300, high=3400):
        nyq = 0.5 * rate
        b, a = butter(2, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, signal)

    def extraer_mfcc(self, signal, rate):
        return mfcc(signal, rate, numcep=13)

    def cargar_modelos(self):
        dirs = [d for d in os.listdir(self.modelos_path) if d.startswith("hmm_")]
        if not dirs:
            self.get_logger().error("No hay modelos entrenados disponibles.")
            return
        dirs.sort(reverse=True)
        modelo_dir = os.path.join(self.modelos_path, dirs[0])
        for archivo in os.listdir(modelo_dir):
            if archivo.endswith(".pkl"):
                nombre = archivo.replace(".pkl", "")
                self.nombres_modelos[nombre] = len(self.nombres_modelos) + 1
                self.modelos[nombre] = joblib.load(os.path.join(modelo_dir, archivo))

    def escuchar_audio(self):
        self.get_logger().info("🎤 Escuchando...")
        audio = sd.rec(int(self.duracion * self.frecuencia), samplerate=self.frecuencia, channels=1, dtype='int16')
        sd.wait()
        signal = audio.flatten()

        signal = self.filtro_pasabanda(signal, self.frecuencia)
        signal = self.preenfasis(signal)
        rms = np.sqrt(np.mean(signal ** 2))
        if rms > 0:
            signal = signal / rms
        mfcc_feat = self.extraer_mfcc(signal, self.frecuencia)

        if len(mfcc_feat) < self.min_frames:
            self.get_logger().warn("Grabación muy corta o ruidosa. Intenta de nuevo.")
            return

        scores = {etiqueta: modelo.score(mfcc_feat) for etiqueta, modelo in self.modelos.items()}
        pred = max(scores, key=scores.get)
        valor = self.nombres_modelos[pred]
        self.get_logger().info(f"🧠 Comando detectado: {pred.upper()} → {valor}")

        msg = Int32()
        msg.data = valor
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MapaComandosNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

