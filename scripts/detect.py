from ultralytics import YOLO
import supervision as sv
import cv2
import os
import numpy as np

model = YOLO("yolov8n.pt")

# Define annotators
ellipse_annotator = sv.EllipseAnnotator()
triangle_annotator = sv.TriangleAnnotator()

class DetectandAnnotate():
    def __init__(self, input_file_path, output_file_path, confidence_threshold=0.5):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO model - YOLOv8 works well for sports detection
        self.model = model  # You can also use yolov8s.pt or yolov8m.pt for better accuracy
        
        # COCO class IDs for football-related objects
        self.PERSON_CLASS_ID = 0  # Person class in COCO dataset
        self.SPORTS_BALL_CLASS_ID = 32  # Sports ball class in COCO dataset
        
        # Define colors for different types of detections
        self.PLAYER_COLOR = sv.Color.BLUE
        self.BALL_COLOR = sv.Color.RED
        
        # Setup annotators with custom styling
        self.setup_annotators()


    def setup_annotators(self):
        """Initialize annotators with custom styling for football detection"""
        
        # Box annotators with different colors for players and ball
        self.player_box_annotator = sv.BoxAnnotator(
            color=self.PLAYER_COLOR,
            thickness=3
        )
        
        self.ball_box_annotator = sv.BoxAnnotator(
            color=self.BALL_COLOR,
            thickness=4
        )
        
        # Label annotators
        self.player_label_annotator = sv.LabelAnnotator(
            color=self.PLAYER_COLOR,
            text_color=sv.Color.WHITE,
            text_scale=0.6,
            text_thickness=2,
            text_padding=5
        )
        
        self.ball_label_annotator = sv.LabelAnnotator(
            color=self.BALL_COLOR,
            text_color=sv.Color.WHITE,
            text_scale=0.8,
            text_thickness=2,
            text_padding=5
        )
        
        # Alternative annotators for variety
        self.player_ellipse_annotator = sv.EllipseAnnotator(
            color=self.PLAYER_COLOR,
            thickness=3
        )
        
        self.ball_circle_annotator = sv.CircleAnnotator(
            color=self.BALL_COLOR,
            thickness=4
        )

    def filter_detections(self, detections):
        """Filter detections to only include players and balls"""
        
        # Get indices for players and balls
        player_indices = detections.class_id == self.PERSON_CLASS_ID
        ball_indices = detections.class_id == self.SPORTS_BALL_CLASS_ID
        
        # Filter confidence threshold
        confidence_mask = detections.confidence >= self.confidence_threshold
        
        # Combine filters
        player_mask = player_indices & confidence_mask
        ball_mask = ball_indices & confidence_mask
        
        # Create separate detection objects
        player_detections = detections[player_mask] if np.any(player_mask) else sv.Detections.empty()
        ball_detections = detections[ball_mask] if np.any(ball_mask) else sv.Detections.empty()
        
        return player_detections, ball_detections
    
    def create_custom_labels(self, detections, object_type="Player"):
        """Create custom labels with confidence scores"""
        if len(detections) == 0:
            return []
        
        labels = []
        for i, confidence in enumerate(detections.confidence):
            if object_type == "Player":
                label = f"Player {i+1}: {confidence:.2f}"
            else:
                label = f"Ball: {confidence:.2f}"
            labels.append(label)
        
        return labels
    
    def annotate_frame(self, frame, player_detections, ball_detections):
        """Annotate frame with player and ball detections"""
        
        annotated_frame = frame.copy()
        
        # Annotate players
        if len(player_detections) > 0:
            player_labels = self.create_custom_labels(player_detections, "Player")
            
            # Use box annotation for players
            annotated_frame = self.player_box_annotator.annotate(
                scene=annotated_frame,
                detections=player_detections
            )
            
            # Add labels
            annotated_frame = self.player_label_annotator.annotate(
                scene=annotated_frame,
                detections=player_detections,
                labels=player_labels
            )
        
        # Annotate balls
        if len(ball_detections) > 0:
            ball_labels = self.create_custom_labels(ball_detections, "Ball")
            
            # Use circle annotation for balls to make them more prominent
            annotated_frame = self.ball_circle_annotator.annotate(
                scene=annotated_frame,
                detections=ball_detections
            )
            
            # Add labels
            annotated_frame = self.ball_label_annotator.annotate(
                scene=annotated_frame,
                detections=ball_detections,
                labels=ball_labels
            )
        
        return annotated_frame
    
    def detect_and_annotate(self):
        """Main detection and annotation pipeline"""
        
        # Validate input file
        if not os.path.exists(self.input_file_path):
            print(f"ERROR: Input file does not exist: {self.input_file_path}")
            return False
        
        print(f"Processing football video: {self.input_file_path}")
        
        # Create output directory if needed
        output_dir = os.path.dirname(self.output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

         # Open video capture
        cap = cv2.VideoCapture(self.input_file_path)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open video file: {self.input_file_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

         # Setup video writer
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_file_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("ERROR: Could not initialize video writer")
            cap.release()
            return False
        
        frame_count = 0
        player_count_total = 0
        ball_count_total = 0
        
        print("Starting detection and annotation...")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run YOLO detection
                results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
                detections = sv.Detections.from_ultralytics(results[0])
                
                # Filter for players and balls
                player_detections, ball_detections = self.filter_detections(detections)
                
                # Count detections
                players_in_frame = len(player_detections)
                balls_in_frame = len(ball_detections)
                
                player_count_total += players_in_frame
                ball_count_total += balls_in_frame
                
                # Annotate frame
                annotated_frame = self.annotate_frame(frame, player_detections, ball_detections)
                
                # Add frame counter and detection info
                info_text = f"Frame: {frame_count}/{total_frames} | Players: {players_in_frame} | Balls: {balls_in_frame}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame
                out.write(annotated_frame)
                
                # Display progress
                if frame_count % 30 == 0:  # Every second at 30fps
                    print(f"Processed {frame_count}/{total_frames} frames - "
                          f"Players: {players_in_frame}, Balls: {balls_in_frame}")
                
                # Optional: Display frame (comment out for faster processing)
                # cv2.imshow("Football Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing interrupted by user")
                    break
        
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return False
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        # Final statistics
        avg_players_per_frame = player_count_total / frame_count if frame_count > 0 else 0
        avg_balls_per_frame = ball_count_total / frame_count if frame_count > 0 else 0
        
        print(f"\nProcessing completed!")
        print(f"Total frames processed: {frame_count}")
        print(f"Average players per frame: {avg_players_per_frame:.1f}")
        print(f"Average balls per frame: {avg_balls_per_frame:.1f}")
        print(f"Output saved to: {self.output_file_path}")
        
        # Verify output file
        if os.path.exists(self.output_file_path):
            output_size = os.path.getsize(self.output_file_path)
            print(f"Output file size: {output_size / (1024*1024):.1f} MB")
            return True
        else:
            print("ERROR: Output file was not created")
            return False

   # def detect_and_annotate(self):
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

def main():
    """Main function to run football detection"""
    
    # Configuration
    input_path = "input_videos/samplevid.mp4"  # Change this to your football video path
    output_path = "output_videos/football_annotated.mp4"
    confidence_threshold = 0.4  # Lower threshold to catch more players
    
    # Create detector instance
    detector = DetectandAnnotate(
        input_file_path=input_path,
        output_file_path=output_path,
        confidence_threshold=confidence_threshold
    )
    
    # Run detection
    success = detector.detect_and_annotate()
    
    if success:
        print("\n Football detection completed successfully!")
        print(f" Check your output video at: {output_path}")
    else:
        print("\n Football detection failed. Check error messages above.")

if __name__ == "__main__":
    main()