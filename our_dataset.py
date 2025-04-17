import os
import pickle
import time
from tqdm import tqdm
import cv2
import numpy as np
import mediapipe as mp
import torch
import random
from glob import glob

def preprocess_entire_dataset(
    video_dir, 
    cache_dir="keypoints_processed",
    max_frames=60
):
    """
    Preprocess and cache an entire dataset of videos organized by gesture class
    
    Args:
        video_dir (str): Path to directory with videos organized by gesture class subfolders
        cache_dir (str): Directory to save processed keypoints
        max_frames (int): Maximum number of frames to process per video
        
    The video_dir should have the following structure:
    video_dir/
        gesture_class_1/
            video1.mp4
            video2.mp4
            ...
        gesture_class_2/
            video1.mp4
            ...
    
    Returns:
        str: Path to the output cache file
    """
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(cache_dir, f"all_keypoints_maxframes_{max_frames}.pkl")
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Loading to check what's already processed...")
        try:
            with open(output_file, 'rb') as f:
                existing_data = pickle.load(f)
            
            # Get IDs of videos that are already processed
            processed_videos = set()
            for item in existing_data:
                video_path = item['video_path']
                processed_videos.add(video_path)
            
            print(f"Found {len(processed_videos)} already processed videos")
        except Exception as e:
            print(f"Error loading existing data: {e}. Starting from scratch.")
            processed_videos = set()
            existing_data = []
    else:
        processed_videos = set()
        existing_data = []
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,  # Set to False for video
        max_num_hands=2,
        min_detection_confidence=0.5,
        model_complexity=1
    )
    
    # Get all gesture classes (subdirectories)
    gesture_classes = [d for d in os.listdir(video_dir) 
                      if os.path.isdir(os.path.join(video_dir, d))]
    
    # Create a mapping from class name to ID
    class_to_id = {class_name: i+1 for i, class_name in enumerate(sorted(gesture_classes))}
    
    print(f"Found {len(gesture_classes)} gesture classes: {gesture_classes}")
    print(f"Class to ID mapping: {class_to_id}")
    
    # Create a list of videos to process
    all_videos = []
    for gesture_class in gesture_classes:
        class_dir = os.path.join(video_dir, gesture_class)
        
        # Get all video files (assuming .mp4 extension, add more if needed)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob(os.path.join(class_dir, f'*{ext}')))
        
        for video_path in video_files:
            # Skip if already processed
            if video_path in processed_videos:
                continue
            
            all_videos.append({
                'video_path': video_path,
                'gesture_class': gesture_class,
                'gesture_id': class_to_id[gesture_class]
            })
    
    print(f"Total videos found: {len(all_videos) + len(processed_videos)}")
    print(f"Videos already processed: {len(processed_videos)}")
    print(f"Videos to process: {len(all_videos)}")
    
    # If everything is already processed, we're done
    if len(all_videos) == 0:
        print("All videos already cached! Dataset is ready.")
        return output_file
    
    # Process videos sequentially with a progress bar
    total_start_time = time.time()
    successful = 0
    failures = 0
    
    # Start with existing data or empty list
    all_data = existing_data if len(processed_videos) > 0 else []
    
    for video_info in tqdm(all_videos, desc="Processing videos"):
        try:
            # Extract information
            video_path = video_info['video_path']
            gesture_class = video_info['gesture_class']
            gesture_id = video_info['gesture_id']
            
            # Get video filename for display purposes
            video_name = os.path.basename(video_path)
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame indices to process
            frames_to_process = min(total_frames, max_frames)
            
            # For longer videos, sample evenly
            if total_frames > max_frames:
                frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
            else:
                frame_indices = list(range(total_frames))
            
            # Process each frame
            frames_keypoints = []
            current_frame = 0
            
            while cap.isOpened() and current_frame < total_frames:
                success, frame = cap.read()
                
                if not success:
                    break
                
                # Only process frames in our selected indices
                if current_frame in frame_indices:
                    # Convert to RGB for MediaPipe
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process with MediaPipe
                    results = hands.process(frame_rgb)
                    
                    # Initialize keypoints tensor (for up to 2 hands)
                    keypoints = np.zeros((2, 21, 3), dtype=np.float32)
                    
                    # Process hand landmarks if detected
                    if results.multi_hand_landmarks:
                        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            if hand_idx >= 2:  # Only consider up to 2 hands
                                break
                            
                            for landmark_idx, landmark in enumerate(hand_landmarks.landmark):
                                keypoints[hand_idx, landmark_idx, 0] = landmark.x
                                keypoints[hand_idx, landmark_idx, 1] = landmark.y
                                keypoints[hand_idx, landmark_idx, 2] = landmark.z
                    
                    frames_keypoints.append(torch.tensor(keypoints))
                
                current_frame += 1
            
            # Release the video
            cap.release()
            
            # If no frames were processed, count as failure
            if len(frames_keypoints) == 0:
                failures += 1
                print(f"Warning: No hand keypoints detected in {video_name}")
                continue
            
            # Stack keypoints into a single tensor
            frames_keypoints_tensor = torch.stack(frames_keypoints)
            
            # Create a data entry with all information
            data_entry = {
                'video_path': video_path,
                'video_name': video_name,
                'gesture_class': gesture_class,
                'gesture_id': gesture_id,
                'keypoints': frames_keypoints_tensor,
                'total_frames': total_frames,
                'processed_frames': len(frames_keypoints)
            }
            
            # Add to our list of data
            all_data.append(data_entry)
            
            successful += 1
            
            # Save intermediate results every 100 successful entries
            if successful % 100 == 0:
                with open(output_file, 'wb') as f:
                    pickle.dump(all_data, f)
                print(f"Saved intermediate results after {successful} videos")
            
        except Exception as e:
            failures += 1
            print(f"Error processing video {video_path}: {str(e)}")
    
    # Close MediaPipe resources
    hands.close()
    
    # Save the final results
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)
    
    # Also save the class mapping for reference
    mapping_file = os.path.join(cache_dir, "class_mapping.pkl")
    with open(mapping_file, 'wb') as f:
        pickle.dump(class_to_id, f)
    
    total_time = time.time() - total_start_time
    print(f"Preprocessing complete!")
    print(f"Successfully processed: {successful} videos")
    print(f"Failed: {failures} videos")
    print(f"Total time: {total_time:.2f} seconds")
    
    if len(all_videos) > 0:
        print(f"Average time per video: {total_time/len(all_videos):.2f} seconds")
    
    # Calculate file size
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Output file size: {file_size:.2f} MB")
    
    # Print information about the processed data
    print(f"Total number of videos in final dataset: {len(all_data)}")
    
    # Count segments by gesture class
    gesture_counts = {}
    for item in all_data:
        gesture_class = item['gesture_class']
        if gesture_class in gesture_counts:
            gesture_counts[gesture_class] += 1
        else:
            gesture_counts[gesture_class] = 1
    
    print("Gesture class distribution:")
    for gesture_class, count in sorted(gesture_counts.items()):
        print(f"  {gesture_class}: {count} videos")
    
    return output_file


class FastCachedGestureDataset(torch.utils.data.Dataset):
    """
    Dataset that loads preprocessed keypoints from a cached file
    """
    def __init__(self, cache_file):
        """
        Args:
            cache_file (str): Path to the cache file containing all the preprocessed data
        """
        # Load the cached data
        with open(cache_file, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} videos from cache")
    
    def __len__(self):
        """Return the number of videos in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns hand keypoints and gesture ID for a single video
        
        Args:
            idx (int): Index of the video
            
        Returns:
            frames_keypoints_tensor (torch.Tensor): Tensor of shape [num_frames, num_hands, num_landmarks, 3]
            gesture_id (int): Gesture ID (1-based indexing)
        """
        item = self.data[idx]
        frames_keypoints_tensor = item['keypoints']
        gesture_id = item['gesture_id']
        
        # Return with 1-based indexing (as provided in the class_to_id mapping)
        return frames_keypoints_tensor, gesture_id
    
    def get_item_info(self, idx):
        """
        Returns detailed information about a video
        
        Args:
            idx (int): Index of the video
            
        Returns:
            dict: Video information without the keypoints tensor
        """
        item = self.data[idx].copy()
        # Remove the keypoints tensor to make the result more readable
        item.pop('keypoints')
        return item
    
    def get_class_distribution(self):
        """
        Returns the distribution of gesture classes in the dataset
        
        Returns:
            dict: Mapping from gesture_class to count
        """
        distribution = {}
        for item in self.data:
            gesture_class = item['gesture_class']
            if gesture_class in distribution:
                distribution[gesture_class] += 1
            else:
                distribution[gesture_class] = 1
        return distribution

    def get_id_distribution(self):
        """
        Returns the distribution of gesture IDs in the dataset
        
        Returns:
            dict: Mapping from gesture_id to count
        """
        distribution = {}
        for item in self.data:
            gesture_id = item['gesture_id']
            if gesture_id in distribution:
                distribution[gesture_id] += 1
            else:
                distribution[gesture_id] = 1
        return distribution


def create_fast_gesture_dataloaders(
    cache_file,
    batch_size=16,
    val_split=0.2,
    num_workers=4,
    max_frames=60,
    random_seed=42
):
    """
    Create training and validation dataloaders for cached gesture keypoint data
    
    Args:
        cache_file (str): Path to the cache file containing all data
        batch_size (int): Batch size for dataloaders
        val_split (float): Proportion of data to use for validation (0.0-1.0)
        num_workers (int): Number of worker processes for data loading
        max_frames (int): Maximum number of frames to sample/pad for each sequence
        random_seed (int): Random seed for reproducibility
        
    Returns:
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Create the dataset
    dataset = FastCachedGestureDataset(cache_file)
    
    # Calculate sizes for the split
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Define collate function
    def gesture_collate_fn(batch, max_frames=max_frames, is_training=True):
        """
        Custom collate function to handle sequences with a fixed maximum frame count
        
        Args:
            batch: List of tuples (keypoints, label)
            max_frames (int): Maximum number of frames to sample/pad
            is_training (bool): Whether in training mode (for frame sampling)
            
        Returns:
            keypoints_batch: Tensor of keypoints with shape (batch_size, max_frames, num_hands, num_landmarks, 3)
            labels: Tensor of labels
        """
        # Get data and labels
        keypoints = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        batch_size = len(keypoints)
        
        # Get dimensions from the first item
        num_hands = keypoints[0].shape[1]
        num_landmarks = keypoints[0].shape[2]
        feature_dim = keypoints[0].shape[3]
        
        # Create tensor for output keypoints
        processed_keypoints = torch.zeros((batch_size, max_frames, num_hands, num_landmarks, feature_dim))
        
        # Process each sequence in the batch
        for i, kp in enumerate(keypoints):
            seq_length = kp.shape[0]
            
            if seq_length > max_frames:
                # If sequence is longer than max_frames, sample frames
                if is_training:
                    # Random sampling during training
                    indices = sorted(random.sample(range(seq_length), max_frames))
                else:
                    # Uniform sampling during validation
                    indices = np.linspace(0, seq_length-1, max_frames, dtype=int)
                
                processed_keypoints[i] = kp[indices]
            else:
                # If sequence is shorter, use all frames and pad with zeros
                processed_keypoints[i, :seq_length] = kp
        
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)
        
        return processed_keypoints, labels
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: gesture_collate_fn(batch, is_training=True)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: gesture_collate_fn(batch, is_training=False)
    )
    
    return train_loader, val_loader


# Example usage
if __name__ == "__main__":
    # Directory structure:
    # video_dir/
    #   gesture_class_1/
    #     video1.mp4
    #     video2.mp4
    #   gesture_class_2/
    #     video1.mp4
    #     ...
    
    video_dir = "path/to/video_directory"
    max_frames = 60
    cache_dir = f"./keypoints_processed_maxframes_{max_frames}"
    
    # Preprocess the dataset
    cache_file = preprocess_entire_dataset(
        video_dir=video_dir, 
        cache_dir=cache_dir, 
        max_frames=max_frames
    )
    
    # Create dataloaders from the cached data
    train_loader, val_loader = create_fast_gesture_dataloaders(
        cache_file=cache_file,
        batch_size=16,
        val_split=0.2,
        max_frames=max_frames,
        num_workers=4
    )
    
    # Print information about the first batch
    for keypoints, labels in train_loader:
        print(f"Batch shape: {keypoints.shape}")
        print(f"Labels: {labels}")
        break