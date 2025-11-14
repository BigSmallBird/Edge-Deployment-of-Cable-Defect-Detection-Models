import gc
import threading
import queue
import time
import numpy as np
import cv2
import onnxruntime as ort

VIDEO_NUMBER = 0
MODEL_PATH = r"/MyENV/project/test1/code/train1/yolo11n.onnx"
POINT_COMUNICATION = {}
CAPISRUNNING = False
track_history = {}

# ONNX Runtime configuration
ORT_SESSION_OPTIONS = ort.SessionOptions()
ORT_SESSION_OPTIONS = ort.SessionOptions()
ORT_SESSION_OPTIONS.intra_op_num_threads = 8  # Match Orange Pi 5's CPU cores
ORT_SESSION_OPTIONS.inter_op_num_threads = 4
ORT_SESSION_OPTIONS.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
ORT_SESSION_OPTIONS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# YOLO configuration
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
NUM_CLASSES = 80  # Default for COCO dataset (adjust if your model has different classes)

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
        self.print_camera_parameter_settings()
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
        self.session = None
        self.input_name = None
        self.output_names = None
        self.first_points = ()
        self.frame = None
        self.boxes = None
        self.confidences = None
        self.class_ids = None
        self.track_ids = None
        self.input_shape = (INPUT_HEIGHT, INPUT_WIDTH)

    def load_model(self, model_path):
        if self.session is None:
            self.session = ort.InferenceSession(
                model_path,
                sess_options=ORT_SESSION_OPTIONS,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            print(f"ONNX model loaded successfully. Input: {self.input_name}, Outputs: {self.output_names}")

    def preprocess_frame(self, frame):
        self.frame = frame
        img = cv2.resize(frame, self.input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, output, original_shape):
        input_h, input_w = self.input_shape
        original_h, original_w = original_shape
        
        boxes = []
        confidences = []
        class_ids = []
        track_ids = []

        for out in output:
            if out.shape[1] == 6:  # YOLO11 output format: [x1, y1, x2, y2, conf, class, track_id] (adjust if needed)
                for detection in out[0]:
                    x1, y1, x2, y2, conf, cls = detection[:6]
                    track_id = detection[6] if len(detection) > 6 else -1
                    
                    if conf > CONF_THRESHOLD:
                        # Rescale boxes to original image size
                        x1 = int(x1 * (original_w / input_w))
                        y1 = int(y1 * (original_h / input_h))
                        x2 = int(x2 * (original_w / input_w))
                        y2 = int(y2 * (original_h / input_h))
                        
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(float(conf))
                        class_ids.append(int(cls))
                        track_ids.append(int(track_id) if track_id != -1 else None)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD
        )
        
        self.boxes = []
        self.confidences = []
        self.class_ids = []
        self.track_ids = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                self.boxes.append(boxes[i])
                self.confidences.append(confidences[i])
                self.class_ids.append(class_ids[i])
                self.track_ids.append(track_ids[i])
        
        self.update_track_history()

    def update_track_history(self):
        global track_history
        if not self.boxes or not self.track_ids:
            return
        
        for i, (box, track_id, conf, cls) in enumerate(zip(
            self.boxes, self.track_ids, self.confidences, self.class_ids
        )):
            if track_id is None or track_id == -1:
                continue
            
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            if track_id not in track_history:
                track_history[track_id] = []
                self.first_points = (center_x, center_y)
                POINT_COMUNICATION[track_id] = self.first_points
            
            track_history[track_id].append((center_x, center_y))
            if len(track_history[track_id]) > 10:
                track_history[track_id].pop(0)

    def infer(self, frame):
        self.load_model(MODEL_PATH)
        preprocessed = self.preprocess_frame(frame)
        outputs = self.session.run(self.output_names, {self.input_name: preprocessed})
        self.postprocess(outputs, frame.shape[:2])

    def get_results(self):
        return {
            'boxes': self.boxes,
            'confidences': self.confidences,
            'class_ids': self.class_ids,
            'track_ids': self.track_ids
        }
    
    def get_frame(self):
        return self.frame 

    def frame_select_detection(self):
        if not self.boxes or self.frame is None:
            return None, None
        
        # Select the first detected object (adjust logic if you need specific object)
        box = self.boxes[0]
        x1, y1, x2, y2 = box
        
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
        return self.class_ids[0] if self.class_ids else None
    
    def get_id(self):
        return self.track_ids[0] if self.track_ids else None

    def get_conf(self):
        return self.confidences[0] if self.confidences else None

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
            cv2.resizeWindow('handle_frame', 800, 600)
            
            # Load ONNX model once
            model.load_model(MODEL_PATH)
            
            while True:
                frame = camera.get_frame()
                if frame is not None:
                    inference_frame = frame
                    if inference_frame.ndim == 2 or (inference_frame.ndim == 3 and inference_frame.shape[2] == 1):
                        inference_frame = cv2.cvtColor(inference_frame, cv2.COLOR_GRAY2BGR)

                    # Run ONNX inference
                    model.infer(inference_frame)
                    location, cropped = model.frame_select_detection()
                    handle_frame = frame.copy()

                    # Draw all detections
                    results = model.get_results()
                    for box, conf, cls_id, track_id in zip(
                        results['boxes'], results['confidences'], results['class_ids'], results['track_ids']
                    ):
                        if box is None:
                            continue
                        x1, y1, x2, y2 = box
                        cv2.rectangle(handle_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"ID:{track_id} Cls:{cls_id} Conf:{conf:.2f}"
                        cv2.putText(handle_frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    if cropped is not None and location is not None:
                        try:
                            detection.set_frame(cropped.copy())
                            diff = detection.morphological_contraction_difference()
                        except Exception as e:
                            diff = None
                            print(f"Defect detection error: {e}")
                        
                        if diff is not None:
                            try:
                                contours, _ = detection.find_contours(diff, min_area=50)
                                x1, y1, x2, y2 = location
                                shifted_contours = []
                                for cnt in contours:
                                    cnt2 = cnt.copy()
                                    cnt2[:, 0, 0] += x1
                                    cnt2[:, 0, 1] += y1
                                    shifted_contours.append(cnt2)
                                
                                current_cls = model.get_cls()
                                current_conf = model.get_conf()
                                current_id = model.get_id()
                                if current_cls is not None and current_conf is not None and current_id is not None:
                                    cv2.putText(
                                        handle_frame,
                                        f"class:{current_cls} confidence:{current_conf:.2f} ID:{current_id}",
                                        (x1, y1),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8,
                                        (255, 255, 0),
                                        2,
                                        cv2.LINE_AA
                                    )
                                handle_frame = cv2.drawContours(handle_frame, shifted_contours, -1, (0, 0, 255), 3)

                            except Exception as e:
                                print(f"Contour drawing error: {e}")
                    
                    cv2.imshow("handle_frame", handle_frame)
                
                key = cv2.waitKey(1) & 0xFF
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
