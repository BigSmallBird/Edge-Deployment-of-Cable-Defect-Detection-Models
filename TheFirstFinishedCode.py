import gc
from re import S
import threading
from webbrowser import get
from cv2.detail import CameraParams
import queue
import time
import numpy as np
import cv2
from ultralytics import YOLO


VIDEO_NUMBER = 1
MODEL_PATH = r"F:\workSpace\test2\code\train5-Spinach\yolo11s-seg.pt"
POINT_COMUNICATION = {}
CAPISRUNNING = False
track_history = {}




class OptimizedCameraCapture:
    def __init__(self):
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = None
        self.width = 0
        self.height = 0
        self.fps = 0

    def setup_camera(self):
        self.cap = cv2.VideoCapture(VIDEO_NUMBER)
        if not self.cap.isOpened():
            print(f"Error opening video stream (index={VIDEO_NUMBER})")
            exit(0)

        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        return True


    
    def print_camera_parameter_settings(self):
        try:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Camera opened index={VIDEO_NUMBER}, width={w}, height={h}, fps={fps}")
            self.width = w
            self.height = h
            self.fps = fps
        except Exception:
            pass
    
    def capture_frames(self):
        global CAPISRUNNING
        while CAPISRUNNING:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                print(f"Failed to grab frame from device index={VIDEO_NUMBER}. Retrying...")
                time.sleep(0.1)
                try:
                    if self.cap is None or not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(VIDEO_NUMBER)
                except Exception:
                    pass
                continue

    def get_latest_frame(self):
        frame = None
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        return frame
    

    def start_capture(self):
        global CAPISRUNNING
        if not self.setup_camera():
            return False
        CAPISRUNNING = True
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        return True
    
    def stop_capture(self):
        global CAPISRUNNING
        CAPISRUNNING = False
        if self.capture_thread is not None:
            self.capture_thread.join()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        gc.collect()
    
    def get_capture(self):
        return self.cap
    
    def get_frame(self):
        return self.get_latest_frame()



class ModelHandle:
    def __init__(self):
        self.model = None
        self.first_points = ()
        self.frame = None
        self.results = None
        self.box = None
        self.conf = None
        self.id = None
        self.cls = None

    
    def load_model(self, model_path, frame):
        global track_history
        self.frame = frame
        if self.model is None:
            self.model = YOLO(model_path)
        self.results = self.model.track(self.frame, persist=True)
        self.box = None
        self.conf = None
        self.id = None
        self.cls = None
        if (
            self.results
            and len(self.results) > 0
            and getattr(self.results[0], 'boxes', None) is not None
            and hasattr(self.results[0].boxes, 'id')
            and self.results[0].boxes.id is not None
            and self.results[0].boxes.xywh is not None
        ):
            boxes = self.results[0].boxes.xywh.cpu()
            track_ids = self.results[0].boxes.id.int().cpu().tolist()
            confs = self.results[0].boxes.conf.cpu().tolist()
            clss = self.results[0].boxes.cls.int().cpu().tolist()
            for box, track_id, conf, cls in zip(boxes, track_ids, confs, clss):
                self.box = box
                self.conf = float(conf)
                self.id = int(track_id)
                self.cls = int(cls)
                x, y, w, h = box.tolist()
                center_x = int(x)
                center_y = int(y)
                if track_id not in track_history:
                    track_history[track_id] = []
                    self.first_points = (center_x, center_y)
                    POINT_COMUNICATION[track_id] = self.first_points
                track_history[track_id].append((center_x, center_y))
                if len(track_history[track_id]) > 10:
                    track_history[track_id].pop(0)
    

    def get_results(self):
        return self.results
    
    def get_frame(self):
        return self.frame 

    def frame_select_detection(self):
        
        if self.box is None or self.frame is None:
            return None, None
        x, y, w, h = self.box.tolist()
        
        cx = int(round(x))
        cy = int(round(y))
        ww = int(round(w))
        hh = int(round(h))
        x1 = cx - ww // 2
        y1 = cy - hh // 2
        x2 = cx + (ww - ww // 2)
        y2 = cy + (hh - hh // 2)
        
        h_img, w_img = self.frame.shape[:2]
        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img, x2))
        y2 = max(0, min(h_img, y2))
        if x2 <= x1 or y2 <= y1:
            return None, None
        cropped = self.frame[y1:y2, x1:x2]
        location = (x1, y1, x2, y2)
        return location, cropped 

    def get_cls(self):
        return self.cls
    
    def get_id(self):
        return self.id

    def get_conf(self):
        return self.conf

    
 

class TranditionnalDefectDetection:

    def __init__(self):
        self.frame = None
        self.binary_frame = None
        self.gray_frame = None
        self.blur_frame = None

    def set_frame(self, frame):
        self.frame = frame
        self.gray_frame = None
        self.blur_frame = None
        self.binary_frame = None


 
    def ensure_gray(self, frame):
        src = frame if frame is not None else self.frame
        if src is None:
            raise ValueError("No frame available for gray conversion")
        if src.ndim == 3 and src.shape[2] == 3:
            self.gray_frame = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_frame = src.copy()
        return self.gray_frame



    def ensure_blur(self, frame: np.ndarray | None = None, ksize: tuple[int, int] = (5, 5)) -> np.ndarray:

        gray = self.ensure_gray(frame)
        self.blur_frame = cv2.GaussianBlur(gray, ksize, 0)
        return self.blur_frame



    def binarize(self, frame: np.ndarray | None = None, thresh: int = 127,
                 maxval: int = 255, method: int = cv2.THRESH_BINARY) -> np.ndarray:

        blur = self.ensure_blur(frame)
        _, self.binary_frame = cv2.threshold(blur, thresh, maxval, method)
        return self.binary_frame



    def transform(self, frame: np.ndarray) -> np.ndarray:

        if frame is None:
            raise ValueError("transform requires a frame input")
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)



    def colorful_threshold(self, frame: np.ndarray | None = None,
                           lower: tuple[int, int, int] = (0, 0, 0),
                           upper: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        
        src = frame if frame is not None else self.frame
        if src is None:
            raise ValueError("No frame available for colorful threshold")
        mask = cv2.inRange(src, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        self.binary_frame = mask
        return mask



    def edge_detection(self, frame: np.ndarray | None = None, low: int = 100, high: int = 200) -> np.ndarray:

        gray = self.ensure_gray(frame)
        edges = cv2.Canny(gray, low, high)
        return edges


    def morphological_contraction_difference(self, frame: np.ndarray | None = None,
                                             kernel_size: tuple[int, int] = (5, 5)) -> np.ndarray:
        bin_img = self.binarize(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        eroded = cv2.erode(bin_img, kernel, iterations=1)
        dilated = cv2.dilate(bin_img, kernel, iterations=1)
        difference = cv2.absdiff(dilated, eroded)
        return difference


    def find_contours(self, binary: np.ndarray | None = None, min_area: int = 50):

        bin_img = binary if binary is not None else self.binary_frame
        if bin_img is None:
            raise ValueError("No binary image available for contour detection")
        contours, hierarchy = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
        return filtered, hierarchy



    def draw_contours(self, image: np.ndarray | None = None, contours=None,
                      color: tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        if image is None:
            # 尝试使用原始帧
            if self.frame is None:
                raise ValueError("No image available to draw contours")
            image = self.frame.copy()
        else:
            image = image.copy()
        if contours is None:
            contours, _ = self.find_contours()
        cv2.drawContours(image, contours, -1, color, thickness)
        return image


    def template_match(self, image: np.ndarray, template: np.ndarray, method=cv2.TM_CCOEFF_NORMED,
                       threshold: float = 0.8):

        if image is None or template is None:
            return []
        res = cv2.matchTemplate(image, template, method)
        loc = np.where(res >= threshold)
        matches = []
        th, tw = template.shape[:2]
        for pt in zip(*loc[::-1]):
            score = float(res[pt[1], pt[0]])
            matches.append((int(pt[0]), int(pt[1]), int(tw), int(th), score))
        return matches


    def get_frame(self) -> np.ndarray | None:
        return self.frame

    def get_gray_frame(self) -> np.ndarray | None:
        return self.gray_frame

    def get_blur_frame(self) -> np.ndarray | None:
        return self.blur_frame



def main():
    camera = OptimizedCameraCapture()
    model = ModelHandle()
    detection = TranditionnalDefectDetection()
    if camera.start_capture():
        try:
            cv2.namedWindow("handle_frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('handle_frame', 800,600)
            while True:
                frame = camera.get_frame()
                if frame is not None:
                    inference_frame = frame
                    if inference_frame.ndim == 2 or (inference_frame.ndim == 3 and inference_frame.shape[2] == 1):
                        inference_frame = cv2.cvtColor(inference_frame, cv2.COLOR_GRAY2BGR)

                    model.load_model(MODEL_PATH, inference_frame)
                    location, cropped = model.frame_select_detection()
                    handle_frame = frame.copy()
                    if cropped is not None and location is not None:
                        try:
                            detection.set_frame(cropped.copy())
                            diff = detection.morphological_contraction_difference()
                        except Exception:
                            diff = None
                        if diff is not None:
                            try:
                                contours, _ = detection.find_contours(diff, min_area=50)
                                x1, y1, x2, y2 = location
                                shifted_contours = []
                                for cnt in contours:
                                    cnt2 = cnt.copy()
                                    cnt2[:, 0, 0] = cnt2[:, 0, 0] + x1
                                    cnt2[:, 0, 1] = cnt2[:, 0, 1] + y1
                                    shifted_contours.append(cnt2)
                                cv2.putText(
                                    handle_frame,
                                    f"class:{model.get_cls()}  confidence:{model.get_conf():.2f}  ID:{model.get_id()}",
                                    (x1, y1),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (255, 255, 0),
                                    2,
                                    cv2.LINE_AA
                                )
                                handle_frame = cv2.drawContours(handle_frame, shifted_contours, -1, (0, 0, 255), 3)

                            except Exception:
                                pass
                    cv2.imshow("handle_frame", handle_frame)
                else:
                    time.sleep(0.01)
                    continue

                key = cv2.waitKey(50) & 0xFF
                if key == ord('q') or key == 27:
                    print("Exiting")
                    break
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            camera.stop_capture()
            cv2.destroyAllWindows()
            print(POINT_COMUNICATION)
            gc.collect()
    else:
        print("Failed to start camera capture")
        exit(0)

if __name__ == "__main__":
    main()