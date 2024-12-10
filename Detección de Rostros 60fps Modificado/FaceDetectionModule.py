import cv2
import mediapipe as mp
import winsound  # Para agregar sonidos al detectar rostros
import csv  # Para el registro en archivo

class FaceDetector:
    def __init__(self, minDetectionCon=0.5, colorRect=(0, 255, 0), colorText=(255, 0, 0)):
        self.minDetectionCon = minDetectionCon
        self.colorRect = colorRect  # Color configurable para el rectángulo
        self.colorText = colorText  # Color configurable para el texto

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

        # Configuración del archivo para guardar las detecciones
        self.filename = "detections.csv"
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Confidence", "Bounding Box X", "Bounding Box Y", "Bounding Box Width", "Bounding Box Height"])

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])

                if draw and detection.score[0] > 0.8:  # Filtrar detecciones con confianza > 80%
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, self.colorText, 2)
                    winsound.Beep(1200, 100)  # Emitir un sonido cuando se detecta un rostro con alta confianza

                    # Registro en archivo
                    self.log_detection(id, detection.score[0], bbox)

        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, self.colorRect, rt)
        # Esquinas superiores
        cv2.line(img, (x, y), (x + l, y), self.colorRect, t)
        cv2.line(img, (x, y), (x, y + l), self.colorRect, t)
        cv2.line(img, (x1, y), (x1 - l, y), self.colorRect, t)
        cv2.line(img, (x1, y), (x1, y + l), self.colorRect, t)
        # Esquinas inferiores
        cv2.line(img, (x, y1), (x + l, y1), self.colorRect, t)
        cv2.line(img, (x, y1), (x, y1 - l), self.colorRect, t)
        cv2.line(img, (x1, y1), (x1 - l, y1), self.colorRect, t)
        cv2.line(img, (x1, y1), (x1, y1 - l), self.colorRect, t)
        return img

    def log_detection(self, id, confidence, bbox):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id, confidence, bbox[0], bbox[1], bbox[2], bbox[3]])


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    pTime = 0
    detector = FaceDetector(minDetectionCon=0.75, colorRect=(0, 255, 0), colorText=(255, 0, 255))

    while True:
        success, img = cap.read()
        if not success:
            break

        img, bboxs = detector.findFaces(img)
        print(f'Rostros detectados: {bboxs}')

        cTime = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 0), 2)
        cv2.imshow("Detección de Rostros", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
