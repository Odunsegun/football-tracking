import cv2
from ultralytics import YOLO
import supervision as sv
import os
import pickle
from supervision import BoxAnnotator
import sys
import numpy as np
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
import pandas as pd

COLORS = {
    "player": (0, 0, 225),
    "referee": (0, 255, 225),
    "ball": (0, 255, 0)
}

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.ball_trail = []
        self.max_trail_length = 10  

        
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
                    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {} ).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns = ['x1', 'y1', 'x2', 'y2'])
        
        #Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions =[{1: {'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }
        
        self.known_referee_ids = set()

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names #class names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            
            
            #convert to supevision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #convert goalkeeper to player object
            #we don't need goalkeeper
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']
            
            #track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                    
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                    self.known_referee_ids.add(track_id)  # Track for exclusion

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}#only one ball, no need for track_id
            #print(detection_supervision)   
            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)    
        return tracks
            
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) # at the bottom of the bounding box
        
        x_center,_ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        #Create an overlay copy
        overlay = frame.copy()

        #Draw the ellipse on the overlay
        cv2.ellipse(
            overlay,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        #Blend the overlay with the original frame
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_width//2) + 15
        y2_rect = (y2 + rectangle_width//2) + 15
        
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
                
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )
        
        return frame
    
    #pointer for the ball
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)
        
        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)
        
        return frame
    
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1050, 750), (1600,870), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total = team_1_num_frames + team_2_num_frames
        if total == 0:
            team_1 = 0.5  # default to equal possession or 0 if you prefer
        else:
            team_1 = team_1_num_frames / total
            
        total = team_1_num_frames + team_2_num_frames
        if total == 0:
            team_2 = 0.5  # default to equal possession or 0 if you prefer
        else:
            team_2 = team_2_num_frames / total

        

        cv2.putText(frame, f"Team 1 Possession: {team_1*100:.2f}%",(1100,800), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Possession: {team_2*100:.2f}%",(1100,850), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        last_ball_pos = None
        last_ball_velocity = None

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num] if frame_num < len(tracks["players"]) else {}
            referee_dict = tracks["referees"][frame_num] if frame_num < len(tracks["referees"]) else {}
            ball_dict = tracks["ball"][frame_num] if frame_num < len(tracks["ball"]) else {}

            # Handle Ball (real or interpolated)
            if ball_dict:
                # Ball detected
                ball_bbox = list(ball_dict.values())[0]['bbox']
                center = get_center_of_bbox(ball_bbox)

                if last_ball_pos is not None:
                    last_ball_velocity = (
                        center[0] - last_ball_pos[0],
                        center[1] - last_ball_pos[1]
                    )
                last_ball_pos = center

                # Add current center to the trail
                self.ball_trail.append(center)
                if len(self.ball_trail) > self.max_trail_length:
                    self.ball_trail.pop(0)

                # Draw the trail (small green dots)
                for point in self.ball_trail[:-1]:
                    cv2.circle(frame, point, 3, (0, 255, 0), -1)

                # Draw the current ball triangle on top
                frame = self.draw_triangle(frame, ball_bbox, COLORS["ball"])
            else:
                # Ball missing, interpolate using previous velocity
                if last_ball_pos and last_ball_velocity:
                    pred_center = (
                        last_ball_pos[0] + last_ball_velocity[0],
                        last_ball_pos[1] + last_ball_velocity[1]
                    )
                    size = 10  # size of fake bbox
                    pred_bbox = [
                        pred_center[0] - size, pred_center[1] - size,
                        pred_center[0] + size, pred_center[1] + size
                    ]

                    frame = self.draw_triangle(frame, pred_bbox, COLORS["ball"])
                    last_ball_pos = pred_center  # but don't accumulate velocity drift

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color", COLORS["player"])
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], COLORS["referee"])

            # Draw team possession
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames



