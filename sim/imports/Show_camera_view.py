import cv2

class ShowCameraView(object):
    def __init__(self, renderer, camName):
        self.renderer = renderer
        self.window = [winName for winName in camName]

    def setParameter(self, widthDic, heightDic, posxDic, posyDic):
        for winName in self.window:
            cv2.namedWindow(winName, 0)
            cv2.resizeWindow(winName, widthDic[winName], heightDic[winName])
            cv2.moveWindow(winName, posxDic[winName], posyDic[winName])

    def show(self, data):
        for winName in self.window:
            self.renderer.update_scene(data, camera=winName)
            pixels = self.renderer.render()
            cv2.imshow(winName, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

class ShowHandCamera(ShowCameraView):
    def __init__(self, renderer, camName):
        super().__init__(renderer, camName)

class ShowHeadCamera(ShowCameraView):
    def __init__(self, renderer, camName):
        super().__init__(renderer, camName)