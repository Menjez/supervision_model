from ultralytics import YOLO
import supervision as sv
import cv2
import os

model = YOLO("yolov9c")

# Define annotators
ellipse_annotator = sv.EllipseAnnotator()
triangle_annotator = sv.TriangleAnnotator()

class DetectandAnnotate():
    def __init__(self, input_file_path, output_file_path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.cv_height = cv2.CAP_PROP_FRAME_HEIGHT
        self.cv_width = cv2.CAP_PROP_FRAME_WIDTH
        self.cv_fps = cv2.CAP_PROP_FPS

    def detect_and_annotate(self):
        # Debug: Check if input file exists
        if not os.path.exists(self.input_file_path):
            print(f"ERROR: Input file does not exist: {self.input_file_path}")
            return False
        
        print(f"Input file found: {self.input_file_path}")
        
        # Debug: Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        input_cap = cv2.VideoCapture(self.input_file_path)
        
        # Debug: Check if video capture opened successfully
        if not input_cap.isOpened():
            print(f"ERROR: Could not open input video: {self.input_file_path}")
            return False
        
        # Get video properties
        width = int(input_cap.get(self.cv_width))
        height = int(input_cap.get(self.cv_height))
        fps = int(input_cap.get(self.cv_fps))
        total_frames = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Debug: Check if properties are valid
        if width == 0 or height == 0 or fps == 0:
            print("ERROR: Invalid video properties (width, height, or fps is 0)")
            input_cap.release()
            return False
        
        # Try different codecs if mp4v doesn't work
        codecs_to_try = ['mp4v', 'XVID', 'H264', 'MJPG']
        output_writer = None
        
        for codec in codecs_to_try:
            fourcc = cv2.VideoWriter.fourcc(*codec)
            output_writer = cv2.VideoWriter(self.output_file_path, fourcc, fps, (width, height))
            
            if output_writer.isOpened():
                print(f"Successfully initialized VideoWriter with codec: {codec}")
                break
            else:
                print(f"Failed to initialize VideoWriter with codec: {codec}")
                output_writer.release()
                output_writer = None
        
        if output_writer is None:
            print("ERROR: Could not initialize VideoWriter with any codec")
            input_cap.release()
            return False
        
        frame_count = 0
        frames_processed = 0
        
        try:
            while input_cap.isOpened():
                ret, frame = input_cap.read()
                if not ret:
                    print(f"End of video or failed to read frame {frame_count}")
                    break
                
                frame_count += 1
                
                # Debug: Print progress every 30 frames
                if frame_count % 100 == 0:
                    print(f"Processing frame {frame_count}/{total_frames}")
                
                try:
                    # Run YOLO prediction
                    results = model.predict(frame, verbose=False)  # Set verbose=False to reduce output
                    detections = sv.Detections.from_ultralytics(results[0])
                    
                    # Annotate frame
                    annotated_frame = ellipse_annotator.annotate(
                        scene=frame.copy(),
                        detections=detections
                    )
                    
                    annotated_frame = triangle_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections
                    )
                    
                    # Write frame to output
                    success = output_writer.write(annotated_frame)
                    if not success and frame_count == 1:
                        print("WARNING: Failed to write first frame - check codec compatibility")
                    
                    frames_processed += 1
                    
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Processing interrupted by user")
                        break
                        
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"General error during processing: {str(e)}")
        
        finally:
            input_cap.release()
            output_writer.release()
            cv2.destroyAllWindows()
            
            print(f"Processing completed: {frames_processed} frames processed")
            
            # Debug: Check if output file was created
            if os.path.exists(self.output_file_path):
                output_size = os.path.getsize(self.output_file_path)
                print(f"Output file created: {self.output_file_path} ({output_size} bytes)")
                if output_size == 0:
                    print("WARNING: Output file is empty")
                return True
            else:
                print(f"ERROR: Output file was not created: {self.output_file_path}")
                return False

if __name__ == "__main__":
    input_path = r"D:\My Learning\LuxDev\supervision\input_videos\samplevid.mp4"
    output_path = "D:\My Learning\LuxDev\supervision\output_videos/annotated_output.mp4"

    print("Starting object detection and annotation...")
    detector = DetectandAnnotate(input_path, output_path)
    success = detector.detect_and_annotate()
    
    if success:
        print("Processing completed successfully!")
    else:
        print("Processing failed - check error messages above")