import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import os
from pathlib import Path

class HandTracker:
    def __init__(self, max_num_hands=10, min_detection_confidence=0.5):
        """Initialize hand tracking with MediaPipe"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_hands(self, frame):
        """Detect hands in frame and return landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hands_data = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get wrist point (landmark 0) as the single tracking point
                wrist = hand_landmarks.landmark[0]
                h, w = frame.shape[:2]
                x, y = int(wrist.x * w), int(wrist.y * h)
                
                # Determine if left or right hand
                hand_label = handedness.classification[0].label
                hand_id = 1 if hand_label == "Left" else 2  # 1 for left, 2 for right
                
                hands_data.append({
                    'x': x,
                    'y': y,
                    'hand_id': hand_id,
                    'confidence': handedness.classification[0].score
                })
                
        return hands_data

class KalmanTracker:
    def __init__(self):
        """Initialize Kalman filter for 2D tracking"""
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        
        # Measurement function
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        
        # Measurement noise
        self.kf.R *= 10
        
        # Process noise
        self.kf.Q *= 0.1
        
        # Initial uncertainty
        self.kf.P *= 100
        
        self.initialized = False

        
    def update(self, x, y):
        """Update Kalman filter with new measurement"""
        if not self.initialized:
            self.kf.x = np.array([x, y, 0, 0])
            self.initialized = True
            
        self.kf.predict()
        self.kf.update(np.array([x, y]))
        
        return self.kf.x[:2]  # Return filtered position
    
    def predict(self, steps=1):
        """Predict future position"""
        if not self.initialized:
            return None
            
        # Save current state
        saved_state = self.kf.x.copy()
        saved_P = self.kf.P.copy()
        
        # Predict future steps
        for _ in range(steps):
            self.kf.predict()
        
        prediction = self.kf.x[:2].copy()
        
        # Restore state
        self.kf.x = saved_state
        self.kf.P = saved_P
        
        return prediction

class PersonTracker:
    def __init__(self, person_id):
        """Track a single person with two hands"""
        self.person_id = person_id
        self.left_tracker = KalmanTracker()
        self.right_tracker = KalmanTracker()
        self.last_left_pos = None
        self.last_right_pos = None
        self.frames_since_seen = 0
        self.data = []
        
    def update(self, hands_data, frame_id):
        """Update person's hand positions"""
        left_hand = None
        right_hand = None
        
        for hand in hands_data:
            if hand['hand_id'] == 1:  # Left hand
                left_hand = hand
            elif hand['hand_id'] == 2:  # Right hand
                right_hand = hand
                
        # Update trackers and store data
        if left_hand:
            filtered_pos = self.left_tracker.update(left_hand['x'], left_hand['y'])
            self.last_left_pos = (left_hand['x'], left_hand['y'])
            self.data.append({
                'frame_id': frame_id,
                'hand_id': 1,
                'x': left_hand['x'],
                'y': left_hand['y'],
                'filtered_x': filtered_pos[0],
                'filtered_y': filtered_pos[1]
            })
            
        if right_hand:
            filtered_pos = self.right_tracker.update(right_hand['x'], right_hand['y'])
            self.last_right_pos = (right_hand['x'], right_hand['y'])
            self.data.append({
                'frame_id': frame_id,
                'hand_id': 2,
                'x': right_hand['x'],
                'y': right_hand['y'],
                'filtered_x': filtered_pos[0],
                'filtered_y': filtered_pos[1]
            })
            
        # Update visibility counter
        if left_hand or right_hand:
            self.frames_since_seen = 0
        else:
            self.frames_since_seen += 1
            
    def predict_future(self, steps=5):
        """Predict future positions"""
        predictions = {}
        
        left_pred = self.left_tracker.predict(steps)
        if left_pred is not None:
            predictions['left'] = left_pred
            
        right_pred = self.right_tracker.predict(steps)
        if right_pred is not None:
            predictions['right'] = right_pred
            
        return predictions
    
    def save_to_csv(self, output_dir):
        """Save person's hand tracking data to CSV"""
        if not self.data:
            return
            
        df = pd.DataFrame(self.data)
        filepath = os.path.join(output_dir, f'{self.person_id}.csv')
        df.to_csv(filepath, index=False)
        print(f"Saved {len(self.data)} records to {filepath}")

class MultiPersonHandTracker:
    def __init__(self, output_dir='hand_tracking_output'):
        """Main class for tracking multiple people's hands"""
        self.hand_tracker = HandTracker()
        self.persons = {}
        self.next_person_id = 1
        self.output_dir = output_dir
        self.max_distance_threshold = 100  # Maximum pixel distance for hand association
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
    def associate_hands_to_persons(self, current_hands, frame_id):
        """Associate detected hands to existing persons or create new ones"""
        if not current_hands:
            return
            
        # Group hands by potential persons (hands close to each other)
        hand_groups = self.group_hands_by_proximity(current_hands)
        
        # Match groups to existing persons
        if self.persons:
            self.match_groups_to_persons(hand_groups, frame_id)
        else:
            # First frame: create persons for each group
            for group in hand_groups:
                person = PersonTracker(self.next_person_id)
                person.update(group, frame_id)
                self.persons[self.next_person_id] = person
                self.next_person_id += 1
                
    def group_hands_by_proximity(self, hands):
        """Group hands that likely belong to the same person"""
        if len(hands) <= 1:
            return [hands]
            
        groups = []
        used = set()
        
        for i, hand1 in enumerate(hands):
            if i in used:
                continue
                
            group = [hand1]
            used.add(i)
            
            # Find nearby opposite hand
            for j, hand2 in enumerate(hands):
                if j in used:
                    continue
                if hand1['hand_id'] == hand2['hand_id']:
                    continue  # Same hand type
                    
                dist = np.sqrt((hand1['x'] - hand2['x'])**2 + 
                             (hand1['y'] - hand2['y'])**2)
                
                if dist < 300:  # Hands of same person typically within 300 pixels
                    group.append(hand2)
                    used.add(j)
                    break
                    
            groups.append(group)
            
        # Add remaining ungrouped hands as individual groups
        for i, hand in enumerate(hands):
            if i not in used:
                groups.append([hand])
                
        return groups
    
    def match_groups_to_persons(self, hand_groups, frame_id):
        """Match hand groups to existing persons using Hungarian algorithm"""
        if not hand_groups:
            return
            
        # Calculate cost matrix
        cost_matrix = np.zeros((len(self.persons), len(hand_groups)))
        person_ids = list(self.persons.keys())
        
        for i, person_id in enumerate(person_ids):
            person = self.persons[person_id]
            for j, group in enumerate(hand_groups):
                cost = self.calculate_group_distance(person, group)
                cost_matrix[i, j] = cost
                
        # Hungarian algorithm for optimal assignment
        if len(person_ids) > 0 and len(hand_groups) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            assigned_groups = set()
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < self.max_distance_threshold:
                    person_id = person_ids[i]
                    self.persons[person_id].update(hand_groups[j], frame_id)
                    assigned_groups.add(j)
                    
            # Create new persons for unassigned groups
            for j, group in enumerate(hand_groups):
                if j not in assigned_groups:
                    person = PersonTracker(self.next_person_id)
                    person.update(group, frame_id)
                    self.persons[self.next_person_id] = person
                    self.next_person_id += 1
                    
        # Update persons not seen in this frame
        for person_id in person_ids:
            if self.persons[person_id].frames_since_seen == 0:
                continue
            self.persons[person_id].frames_since_seen += 1
            
            # Remove person if not seen for many frames
            if self.persons[person_id].frames_since_seen > 30:
                self.persons[person_id].save_to_csv(self.output_dir)
                del self.persons[person_id]
    
    def calculate_group_distance(self, person, group):
        """Calculate distance between person's last position and hand group"""
        total_dist = 0
        count = 0
        
        for hand in group:
            if hand['hand_id'] == 1 and person.last_left_pos:
                dist = np.sqrt((hand['x'] - person.last_left_pos[0])**2 + 
                             (hand['y'] - person.last_left_pos[1])**2)
                total_dist += dist
                count += 1
            elif hand['hand_id'] == 2 and person.last_right_pos:
                dist = np.sqrt((hand['x'] - person.last_right_pos[0])**2 + 
                             (hand['y'] - person.last_right_pos[1])**2)
                total_dist += dist
                count += 1
                
        if count == 0:
            return float('inf')
            
        return total_dist / count
    
    def process_video(self, video_path, visualize=True):
        """Process entire video file"""
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Optional: Setup video writer for output visualization
        if visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter('output_tracked.mp4', fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect hands
            hands_data = self.hand_tracker.detect_hands(frame)
            
            # Associate to persons and track
            self.associate_hands_to_persons(hands_data, frame_id)
            
            # Visualize if requested
            if visualize:
                vis_frame = self.visualize_frame(frame, frame_id)
                cv2.imshow('Hand Tracking', vis_frame)
                out_video.write(vis_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            frame_id += 1
            
            if frame_id % 30 == 0:
                print(f"Processed {frame_id} frames, tracking {len(self.persons)} persons")
                
        # Save remaining data
        for person in self.persons.values():
            person.save_to_csv(self.output_dir)
            
        cap.release()
        if visualize:
            out_video.release()
            cv2.destroyAllWindows()
            
        print(f"Processing complete! Data saved to {self.output_dir}/")
        
    def visualize_frame(self, frame, frame_id):
        """Draw tracking visualization on frame"""
        vis_frame = frame.copy()
        
        for person_id, person in self.persons.items():
            # Draw current positions
            if person.last_left_pos and person.frames_since_seen == 0:
                cv2.circle(vis_frame, person.last_left_pos, 10, (0, 255, 0), -1)
                cv2.putText(vis_frame, f"P{person_id}-L", 
                           (person.last_left_pos[0]-20, person.last_left_pos[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                           
            if person.last_right_pos and person.frames_since_seen == 0:
                cv2.circle(vis_frame, person.last_right_pos, 10, (255, 0, 0), -1)
                cv2.putText(vis_frame, f"P{person_id}-R",
                           (person.last_right_pos[0]-20, person.last_right_pos[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                           
            # Draw predictions
            predictions = person.predict_future(steps=10)
            if 'left' in predictions:
                pred_pos = predictions['left']
                cv2.circle(vis_frame, (int(pred_pos[0]), int(pred_pos[1])), 
                          5, (0, 128, 0), -1)
                          
            if 'right' in predictions:
                pred_pos = predictions['right']
                cv2.circle(vis_frame, (int(pred_pos[0]), int(pred_pos[1])), 
                          5, (128, 0, 0), -1)
                          
        # Add frame counter
        cv2.putText(vis_frame, f"Frame: {frame_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                   
        return vis_frame

def main():
    """Main function to run the hand tracking system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-person hand tracking with Kalman filter')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default='hand_tracking_output',
                       help='Directory to save CSV files (default: hand_tracking_output)')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Create tracker and process video
    tracker = MultiPersonHandTracker(output_dir=args.output_dir)
    tracker.process_video(args.video_path, visualize=not args.no_visualize)
    
    # Print summary
    print("\n=== Tracking Summary ===")
    print(f"Output directory: {args.output_dir}")
    print(f"CSV files created: {len(os.listdir(args.output_dir))}")
    
    # Display sample predictions
    print("\n=== Sample Predictions ===")
    for person_id, person in tracker.persons.items():
        predictions = person.predict_future(steps=30)  # Predict 1 second ahead at 30fps
        if predictions:
            print(f"Person {person_id} predictions (30 frames ahead):")
            for hand, pos in predictions.items():
                print(f"  {hand} hand: x={pos[0]:.1f}, y={pos[1]:.1f}")

if __name__ == "__main__":
    main()