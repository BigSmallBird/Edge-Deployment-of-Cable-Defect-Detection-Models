import os
import glob
import tempfile
from rknn.api import RKNN
import numpy as np
import shutil


def collect_calib_dataset(calib_dir, out_txt_path=None, max_images=200):
    if not calib_dir:
        return None
    if not os.path.isdir(calib_dir):
        return None

    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp"]
    imgs = []
    for p in patterns:
        imgs.extend(glob.glob(os.path.join(calib_dir, p), recursive=True))

    if len(imgs) == 0:
        return None

    imgs = imgs[:max_images]
    if out_txt_path is None:
        fd, out_txt_path = tempfile.mkstemp(prefix="rknn_calib_", suffix=".txt")
        os.close(fd)

    with open(out_txt_path, "w") as f:
        for p in imgs:
            f.write(p + "\n")

    return out_txt_path


def onnx_to_rknn(onnx_path, rknn_path, target_platform="rk3588", calib_dir=None):
    rknn = RKNN(verbose=True)

    print("Step 1/4: Configuring RKNN model...")
    ret = rknn.config(
        target_platform=target_platform,
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        quantized_dtype="w8a8",
        optimization_level=0,
    )
    if ret != 0:
        print("Failed to configure RKNN model!")
        rknn.release()
        return False

    print(f"Step 2/4: Loading ONNX model: {onnx_path}")
    def try_simplify_model(src_path):
        try:
            import onnx
            from onnxsim import simplify
        except Exception:
            return None

        out_path = None
        try:
            print("Attempting ONNX shape inference and simplification...")
            m = onnx.load(src_path)
            try:
                m = onnx.shape_inference.infer_shapes(m)
            except Exception:
                pass
            m_simp, check = simplify(m)
            if not check:
                print("ONNX simplifier check failed, skipping simplified model.")
                return None
            fd, out_path = tempfile.mkstemp(prefix="onnx_simp_", suffix=".onnx")
            os.close(fd)
            onnx.save(m_simp, out_path)
            print(f"Simplified ONNX saved to: {out_path}")
            return out_path
        except Exception as e:
            print("ONNX simplification failed:", e)
            if out_path and os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass
            return None

    simp_onnx = try_simplify_model(onnx_path)
    model_to_load = simp_onnx if simp_onnx is not None else onnx_path

    ret = rknn.load_onnx(model=model_to_load)
    if ret != 0:
        if simp_onnx is not None and model_to_load != onnx_path:
            print("Loading simplified model failed, retrying with original ONNX...")
            ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            print("Failed to load ONNX model!")
            rknn.release()
            if simp_onnx and os.path.exists(simp_onnx):
                try:
                    os.remove(simp_onnx)
                except Exception:
                    pass
            return False

    print("Step 3/4: Building RKNN model...")

    def try_fix_reshape_model(src_path):
        try:
            import onnx
        except Exception:
            return None

        try:
            print("Attempting to fix dynamic Reshape nodes in ONNX model...")
            m = onnx.load(src_path)
            try:
                m = onnx.shape_inference.infer_shapes(m)
            except Exception:
                pass

            graph = m.graph
            init_names = {init.name for init in graph.initializer}
            value_shape = {}
            for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
                if vi.type.HasField('tensor_type'):
                    shape = []
                    for d in vi.type.tensor_type.shape.dim:
                        if d.HasField('dim_value'):
                            shape.append(int(d.dim_value))
                        else:
                            shape.append(None)
                    value_shape[vi.name] = shape

            modified = False
            for node in graph.node:
                if node.op_type == 'Reshape' and len(node.input) >= 2:
                    shape_input = node.input[1]
                    if shape_input not in init_names:
                        out_name = node.output[0]
                        out_shape = value_shape.get(out_name)
                        if out_shape and all([s is not None for s in out_shape]):
                            import numpy as _np
                            shape_arr = _np.array(out_shape, dtype=_np.int64)
                            new_name = shape_input + '_const_shape'
                            from onnx import helper, TensorProto
                            init_tensor = helper.make_tensor(name=new_name,
                                                             data_type=TensorProto.INT64,
                                                             dims=shape_arr.shape,
                                                             vals=shape_arr.flatten().tolist())
                            graph.initializer.append(init_tensor)
                            node.input[1] = new_name
                            init_names.add(new_name)
                            modified = True

            if not modified:
                print("No dynamic Reshape patterns found or unable to infer shapes.")
                return None

            fd, out_path = tempfile.mkstemp(prefix='onnx_fix_reshape_', suffix='.onnx')
            os.close(fd)
            onnx.save(m, out_path)
            print(f"Fixed ONNX saved to: {out_path}")
            return out_path
        except Exception as e:
            print("Reshape fix failed:", e)
            return None

    current_onnx = model_to_load
    def safe_build(do_quantization=False, dataset=None):
        nonlocal current_onnx
        try:
            if dataset is not None:
                ret = rknn.build(do_quantization=do_quantization, dataset=dataset)
            else:
                ret = rknn.build(do_quantization=do_quantization)
            return ret
        except Exception as e:
            msg = str(e)
            print("Build raised exception:", msg)
            if 'Reshape' in msg or 'NOT_IMPLEMENTED' in msg:
                fixed = try_fix_reshape_model(current_onnx)
                if fixed:
                    print("Retrying build with fixed ONNX model...")
                    try:
                        rknn.release()
                    except Exception:
                        pass
                    rknn2 = RKNN(verbose=True)
                    rknn2.config(target_platform=target_platform,
                                 mean_values=[[0,0,0]], std_values=[[255,255,255]],
                                 quantized_dtype='w8a8', optimization_level=0)
                    ret = rknn2.load_onnx(model=fixed)
                    if ret != 0:
                        print("Failed to load fixed ONNX model into new RKNN instance")
                        return -1
                    try:
                        if dataset is not None:
                            ret = rknn2.build(do_quantization=do_quantization, dataset=dataset)
                        else:
                            ret = rknn2.build(do_quantization=do_quantization)
                    except Exception as e2:
                        print("Retry build still failed:", e2)
                        try:
                            rknn2.release()
                        except Exception:
                            pass
                        return -1
                    try:
                        if ret == 0:
                            rknn2.export_rknn(rknn_path)
                    finally:
                        try:
                            rknn2.release()
                        except Exception:
                            pass
                    return ret
            raise

    # If calibration images are provided, create a dataset txt and run quantization.
    if calib_dir:
        print(f"Looking for calibration images in: {calib_dir}")
        calib_txt = collect_calib_dataset(calib_dir)
        if calib_txt:
            print(f"Found calibration images. Using list: {calib_txt}")
            ret = safe_build(do_quantization=True, dataset=calib_txt)
            if ret != 0:
                print("Failed to build RKNN model with quantization!")
                rknn.release()
                return False
        else:
            print("No calibration images found in the provided calib_dir.\n"
                  "Falling back to building without quantization.\n"
                  "For best results on rk3588, provide representative images in calib_dir and retry.")
            ret = safe_build(do_quantization=False)
            if ret != 0:
                print("Failed to build RKNN model (non-quantized)!")
                rknn.release()
                return False
    else:
        print("No calib_dir provided. Building without quantization (FP model).\n"
              "If you want an int8-like quantized model for rk3588, supply a calib_dir with images.")
        ret = safe_build(do_quantization=False)
        if ret != 0:
            print("Failed to build RKNN model (non-quantized)!")
            rknn.release()
            return False

    print(f"Step 4/4: Exporting RKNN model to: {rknn_path}")
    # If safe_build's retry already exported the RKNN file, skip exporting
    if os.path.exists(rknn_path):
        print(f"RKNN file {rknn_path} already exists (likely exported by retry). Skipping export.")
    else:
        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            print("Failed to export RKNN model!")
            rknn.release()
            return False

    rknn.release()
    print("\n✅ ONNX to RKNN conversion completed successfully!")
    return True

if __name__ == "__main__":
    onnx_model_path = "/root/Myenv/TransformText/yolo11n.onnx"
    rknn_model_path = "/root/Myenv/TransformText/yolo11n_rk3588.rknn"
    # Auto-detect a common calibration images folder on the device. If you
    # have calibration images, put them under /root/calib_images/ (jpg/png).
    calib_dir = "/root/calib_images"
    if not os.path.isdir(calib_dir):
        print(f"Calibration directory {calib_dir} not found. Building without quantization by default.")
        calib_dir = None

    onnx_to_rknn(onnx_model_path, rknn_model_path, target_platform="rk3588", calib_dir=calib_dir)
    
