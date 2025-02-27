import depthai as dai
import cv2
import time

def create_pipeline():
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(1920, 1080)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)
    return pipeline
def list_available_devices():
    # Get a list of connected devices
    devices = dai.Device.getAllAvailableDevices()
    if devices:
        print("Available OAK Devices:")
        for idx, device in enumerate(devices):
            print(f"{idx + 1}. Device MxId: {device.getMxId()} - {device.state.name}")
    else:
        print("No OAK devices found. Please check your connection.")

def main():
    list_available_devices()
  
    pipeline = create_pipeline()
    attempt = 0
    max_attempts = 5

    while attempt < max_attempts:
        try:
            # Attempt to connect to the device and start the pipeline
            with dai.Device(pipeline) as device:
                q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
                print("Capturing image...")
                frame = q_rgb.get()
                img = frame.getCvFrame()
                cv2.imwrite("output.jpg", img)
                print("Image saved as output.jpg")
                cv2.imshow("Captured Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break  # Break the loop if successful
        except RuntimeError as e:
            print(f"Failed to connect to device: {e}")
            attempt += 1
            time.sleep(1)  # Wait a bit before retrying

        if attempt == max_attempts:
            print("Failed to connect to the OAK device after several attempts.")

if __name__ == "__main__":
    main()
