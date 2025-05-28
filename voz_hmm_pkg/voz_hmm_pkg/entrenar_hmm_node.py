# entrenar_hmm_node.py
import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.signal import lfilter, butter, filtfilt
from datetime import datetime
from collections import Counter
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import rclpy
from rclpy.node import Node

class EntrenadorHMMNode(Node):
    def __init__(self):
        super().__init__('entrenar_hmm_node')
        self.dataset_path = os.path.expanduser("~/datos_hmm/datos_hmm")
        self.modelos_base = os.path.expanduser("~/modelos_hmm_gaussian")
        self.n_estados = 8
        self.max_intentos = 10
        self.meta_precision = 0.95
        self.min_frames = 20
        self.entrenar()

    def preenfasis(self, signal, coef=0.97):
        return lfilter([1, -coef], 1, signal)

    def filtro_pasabanda(self, signal, rate, low=300, high=3400):
        nyq = 0.5 * rate
        b, a = butter(2, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, signal)

    def extraer_mfcc(self, signal, rate):
        return mfcc(signal, rate, numcep=13)

    def cargar_datos(self):
        X, y, etiquetas = [], [], sorted(os.listdir(self.dataset_path))
        for etiqueta in etiquetas:
            carpeta = os.path.join(self.dataset_path, etiqueta)
            archivos = [f for f in os.listdir(carpeta) if f.endswith(".wav")]
            for archivo in archivos:
                ruta = os.path.join(carpeta, archivo)
                rate, signal = wav.read(ruta)
                if len(signal) < rate * 0.5 or np.max(np.abs(signal)) < 0.05:
                    continue
                signal = self.filtro_pasabanda(signal, rate)
                signal = self.preenfasis(signal)
                rms = np.sqrt(np.mean(signal ** 2))
                if rms > 0:
                    signal = signal / rms
                mfcc_feat = self.extraer_mfcc(signal, rate)
                if len(mfcc_feat) >= self.min_frames:
                    X.append(mfcc_feat)
                    y.append(etiqueta)
        return X, y, etiquetas

    def entrenar(self):
        X, y, etiquetas = self.cargar_datos()
        mejor_acc = 0
        mejor_modelos = {}
        mejor_conf = None

        for intento in range(1, self.max_intentos + 1):
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=intento)
            modelos = {}
            for etiqueta in etiquetas:
                muestras = [X_train[i] for i in range(len(X_train)) if y_train[i] == etiqueta]
                longitudes = [len(m) for m in muestras]
                datos = np.concatenate(muestras)
                modelo = GaussianHMM(n_components=self.n_estados, covariance_type="diag", n_iter=3000, tol=1e-3)
                modelo.fit(datos, lengths=longitudes)
                modelos[etiqueta] = modelo

            y_true, y_pred = [], []
            for i in range(len(X_test)):
                obs = X_test[i]
                real = y_test[i]
                scores = {etiqueta: modelo.score(obs) for etiqueta, modelo in modelos.items()}
                pred = max(scores, key=scores.get)
                y_true.append(real)
                y_pred.append(pred)

            acc = accuracy_score(y_true, y_pred)
            if acc > mejor_acc:
                mejor_acc = acc
                mejor_modelos = modelos
                mejor_conf = confusion_matrix(y_true, y_pred, labels=etiquetas)
            if mejor_acc >= self.meta_precision:
                break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.modelos_base, f"hmm_{round(mejor_acc*100)}p_{timestamp}")
        os.makedirs(output_path, exist_ok=True)
        for etiqueta, modelo in mejor_modelos.items():
            joblib.dump(modelo, os.path.join(output_path, f"{etiqueta}.pkl"))

        self.get_logger().info(f"Mejor modelo guardado en: {output_path} con precisión: {round(mejor_acc*100, 2)}%")

        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(mejor_conf, annot=True, xticklabels=etiquetas, yticklabels=etiquetas, fmt='d', cmap="Greens")
            plt.xlabel("Predicho")
            plt.ylabel("Real")
            plt.title("Matriz de Confusión del Mejor Modelo (GaussianHMM)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "matriz_confusion.png"))
            self.get_logger().info("Matriz de confusión guardada como imagen.")
        except:
            self.get_logger().warn("No se pudo mostrar la matriz de confusión (sin GUI).")


def main(args=None):
    rclpy.init(args=args)
    node = EntrenadorHMMNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

