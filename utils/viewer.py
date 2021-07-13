import uuid
import cv2


class OpenCVImageViewer:
    """
    Rendering viewer for dm_control environments.
    """
    def __init__(self, *, escape_to_exit=False):
        self._escape_to_exit = escape_to_exit
        self._window_name = str(uuid.uuid4())
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        self._isopen = True

    def __del__(self):
        cv2.destroyWindow(self._window_name)
        self._isopen = False

    def imshow(self, img):
        cv2.imshow(self._window_name, img[:, :, [2, 1, 0]])
        if cv2.waitKey(1) in [27] and self._escape_to_exit:
            exit()

    @property
    def isopen(self):
        return self._isopen

    def close(self):
        pass
