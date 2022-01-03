import numpy as np
import cv2
import os

MainProperties = {
    "ferramenta": {
        "id": -1,
        "index": 0,
        "save_preview": False,
        "pixels": 0,
        "edgeDetector": False
    }
}

MainFilters = {
    "gray": False,
    "roi": False,
    "blur": False,
    "edges": {
        "enable": False,
        "kernelx": 3,
        "kernely": 3,
        "kernelFx": 3,
        "kernelFy": 3,
        "thresh1": 127,
        "thresh2": 255,
        "external": True,
        "internal": False
    },
    "pixels": False,
}

address = "https://192.168.1.2:8080/video"


class vision:
    def __init__(self):
        # Inicia a câmera
        self.video = cv2.VideoCapture(address)
        # Seta a resolução para 4K
        self.video.set(3, 1920)
        self.video.set(4, 1080)
        self.roi_img = []
        self.final_img = []
        self.raw_img = []

    def updateValues(self):
        self.blur = MainFilters["blur"]
        self.roi = MainFilters["roi"]
        self.edges = MainFilters["edges"]
        self.pixels = MainFilters["pixels"]

    def rgb_to_hsv(self, r, g, b):
        r, g, b = r/255.0, g/255.0, b/255.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx-mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g-b)/df) + 360) % 360
        elif mx == g:
            h = (60 * ((b-r)/df) + 120) % 360
        elif mx == b:
            h = (60 * ((r-g)/df) + 240) % 360
        s = 0 if mx == 0 else (df/mx)*100
        v = mx*100
        return (h, s, v)

    def countPixels(self, img):
        x, y = self.pixels
        x = int((img.shape[1] * x) / 720)
        y = int((img.shape[0] * y) / 480)
        bgr_value = img[y, x]
        rgb = tuple(reversed(bgr_value))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = (int(rgb[0] - (rgb[0] * .25)), int(rgb[1] -
                 (rgb[1] * .25)), int(rgb[2] - (rgb[2] * .25)))
        upper = (int(rgb[0] + (rgb[0] * .25)), int(rgb[1] +
                 (rgb[1] * .25)), int(rgb[2] + (rgb[2] * .25)))

        Hl, Sl, Vl = self.rgb_to_hsv(lower[0], lower[1], lower[2])
        Hu, Su, Vu = self.rgb_to_hsv(upper[0], upper[1], upper[2])

        lower = (Hl, Sl, Vl)
        upper = (Hu, Su, Vu)

        mask1 = cv2.inRange(img, lower, upper)
        return np.sum(mask1)

    def apply_blur(self, img):
        blur = int(float(self.blur))
        ksize = int(blur) if int(blur) % 2 == 1 else int(blur) + 1
        blur = cv2.GaussianBlur(
            img, (int(ksize), int(ksize)), 0)
        return blur.copy()

    def reduceNoiseAndDetect(self, img):
        kernelX = int(MainFilters["edges"]["kernelx"])
        kernelY = int(MainFilters["edges"]["kernely"])
        kernelFX = int(MainFilters["edges"]["kernelFx"])
        kernelFY = int(MainFilters["edges"]["kernelFy"])
        kernel = tuple((kernelX, kernelY))
        kernelF = tuple((kernelFX, kernelFY))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        filtro = cv2.filter2D(img, -1, kernelF)
        gradient = cv2.morphologyEx(filtro, cv2.MORPH_GRADIENT, kernel)
        opening = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(
            closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        return (contours, hierarchy)

    def getROI(self, img):
        x, y, w, h = self.roi
        # Transforma de string para inteiro
        x = int(float(x))
        y = int(float(y))
        w = int(float(w))
        h = int(float(h))

        # Converte para a resolução correta
        x = int((1920 * x) / 720)
        y = int((1080 * y) / 480)
        w = int((1920 * w) / 720)
        h = int((1080 * h) / 480)

        # Seleciona o Ridge of Interest da imagem
        ROI = img[y:y+h, x:x+w]
        return ROI.copy()

    def matchContours(self, cnts1, cnts2):
        if len(cnts1) != len(cnts2):
            return 2
        total = 0
        for i in range(len(cnts1)):
            matches = cv2.matchShapes(cnts1[i], cnts2[i], 1, 0.0)
            if matches <= 0.15:
                total += 1
        return total > len(cnts1) * .6

    def edgeDetection(self, final):
        contours, hierarchy = self.reduceNoiseAndDetect(final)
        sample = cv2.imread(
            f"static/imgs/{MainProperties['ferramenta']['id']}/1.jpeg")
        sample, _ = self.reduceNoiseAndDetect(sample)
        cnt1 = np.asarray(sample, dtype=object)
        cnt2 = np.asarray(contours, dtype=object)
        matches = self.matchContours(cnt1, cnt2)
        for i in range(len(contours)):
            # Internal = !=
            # External = ==
            if self.edges["external"] and hierarchy[0][i][3] == -1:
                if matches:
                    MainProperties["ferramenta"]["edgeDetector"] = True
                    cv2.drawContours(final, contours, i, (0, 255, 0), 1)
                else:
                    MainProperties["ferramenta"]["edgeDetector"] = False
                    cv2.drawContours(final, contours, i, (0, 0, 255), 1)
            if self.edges["internal"] and hierarchy[0][i][3] != -1:
                if matches:
                    MainProperties["ferramenta"]["edgeDetector"] = True
                    cv2.drawContours(final, contours, i, (0, 255, 0), 1)
                else:
                    MainProperties["ferramenta"]["edgeDetector"] = False
                    cv2.drawContours(final, contours, i, (0, 0, 255), 1)
        return final

    def raw(self):
        if self.raw_img != []:
            _, jpeg = cv2.imencode('.jpg', self.raw_img)
            return jpeg.tobytes()
        img = cv2.imread("static/imgs/00.jpg")
        _, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def view(self):  # sourcery skip: class-extract-method
        self.updateValues()
        print(self.final_image)
        if self.final_img == []:
            img = cv2.imread("static/imgs/00.jpg")
            _, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()
        img = self.final_img
        final = img.copy()
        if self.roi:
            ROI = self.getROI(final)
            self.roi_img = ROI.copy()
            final = ROI.copy()
            if float(self.blur) > 0:
                final = self.apply_blur(final)

            if self.edges["enable"]:
                final = self.edgeDetection(final)

            if self.pixels:
                num = self.countPixels(final)
                MainProperties["ferramenta"]["pixels"] = num
        _, jpeg = cv2.imencode('.jpg', final)
        self.final_img = []
        return jpeg.tobytes()

    def trigger(self):
        _, img = self.video.read()
        self.final_img = img.copy()

    def savePreview(self, img):
        if int(MainProperties["ferramenta"]["id"]) > 0:
            if not os.path.exists(f"static/imgs/{MainProperties['ferramenta']['id']}"):
                os.mkdir(f"static/imgs/{MainProperties['ferramenta']['id']}")
            cv2.imwrite(
                f"static/imgs/{MainProperties['ferramenta']['id']}/1.jpeg", img)

    def preview(self):
        self.updateValues()
        _, img = self.video.read()
        self.raw_img = img
        final = img.copy()
        if self.roi:
            ROI = self.getROI(final)
            self.roi_img = ROI.copy()

            final = ROI.copy()
            if float(self.blur) > 0:
                final = self.apply_blur(final)
            if self.edges["enable"]:
                contours, hierarchy = self.reduceNoiseAndDetect(final)
                for i in range(len(contours)):
                    # Internal = !=
                    # External = ==
                    if self.edges["external"] and hierarchy[0][i][3] == -1:
                        cv2.drawContours(final, contours, i, (0, 255, 0), 1)
                    if self.edges["internal"] and hierarchy[0][i][3] != -1:
                        cv2.drawContours(final, contours, i, (0, 255, 0), 1)

            if self.pixels:
                num = self.countPixels(final)
                MainProperties["ferramenta"]["pixels"] = num

            if MainProperties["ferramenta"]["save_preview"]:
                MainProperties["ferramenta"]["save_preview"] = False
                self.savePreview(ROI)

        _, jpeg = cv2.imencode('.jpg', final)
        return jpeg.tobytes()
