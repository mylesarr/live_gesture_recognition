import cv2
import os
from datetime import datetime
import time
import platform
import subprocess
import threading

def play_beep():
    """
    Play a system beep sound based on the operating system
    """
    if platform.system() == 'Darwin':  # macOS
        subprocess.run(['afplay', '/System/Library/Sounds/Ping.aiff'])
    elif platform.system() == 'Linux':
        subprocess.run(['paplay', '/usr/share/sounds/freedesktop/stereo/message.oga'])
    elif platform.system() == 'Windows':
        import winsound
        winsound.Beep(1000, 500)  # 1000 Hz for 500ms

def play_beep_async():
    """
    Play the beep sound in a separate thread
    """
    thread = threading.Thread(target=play_beep)
    thread.start()
    return thread

def countdown_timer(seconds, window_name='Countdown'):
    """
    Display a visual countdown timer
    
    Args:
        seconds (int): Number of seconds to count down
        window_name (str): Name of the preview window
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 3
    
    # Get camera feed for background
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera for countdown")
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    for i in range(seconds, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add countdown text
        text = str(i)
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (frame_height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, 
                   font_scale, (0, 255, 255), font_thickness)
        
        cv2.imshow(window_name, frame)
        
        # If it's the last second, play the beep 500ms before the end
        if i == 1:
            time.sleep(0.5)  # Wait for first half of the last second
            play_beep_async()  # Play beep
            time.sleep(0.5)  # Wait for remaining time
        else:
            cv2.waitKey(1000)  # Wait 1 second
    
    cap.release()

def capture_consecutive_videos(num_clips, output_dir="/Users/mastermyles/sapiens/Training Videos", warmup_time=5):
    """
    Capture n consecutive 3-second video clips using the default camera
    
    Args:
        num_clips (int): Number of consecutive clips to record
        output_dir (str): Directory to store the video clips
        warmup_time (int): Number of seconds to wait before starting recording
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Standard fps for webcam
    
    # Warm-up period with countdown
    print(f"\nStarting in {warmup_time} seconds...")
    cap.release()  # Release camera for countdown
    countdown_timer(warmup_time, "Preparing to record...")
    
    # Reinitialize camera after countdown
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not reopen camera after warmup")
        return
    
    for clip_num in range(num_clips):
        # Create video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"clip_{clip_num + 1}_{timestamp}.mp4")
        
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        print(f"\nRecording clip {clip_num + 1}/{num_clips}...")
        
        time.sleep(1)
        beep_thread = play_beep_async()
        time.sleep(.25)

        # Record for 2 seconds
        start_time = time.time()
        end_time = start_time + 2
        
        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            # Write the frame
            out.write(frame)
            
            # Display the frame
            cv2.imshow('Recording...', frame)
            
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nRecording stopped by user")
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                return
        
        # Release the video writer for this clip
        out.release()
        print(f"Clip {clip_num + 1} saved to: {output_path}")
        
        # Small delay between clips (slightly longer to account for beep)
        if clip_num < num_clips - 1:
            print("Preparing for next clip...")
            time.sleep(0.5)  # Shorter delay since we'll wait for beep anyway
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("\nAll clips recorded successfully!")

if __name__ == "__main__":
    # Example usage: capture 3 consecutive clips with 5 second warmup
    num_clips = 3
    capture_consecutive_videos(num_clips, warmup_time=3)
