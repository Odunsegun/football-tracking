from utils import read_video, save_video, get_bbox_width, get_center_of_bbox, get_foot_position, measure_distance, measure_xy_distance
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
import numpy as np
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    #read video
    video_frames = read_video('input_videos/Spain_vs_France 2024.mp4')
    
    #Initialize Tracker
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True, 
                                       stub_path='stubs/track_stubs.pkl')
    
    #get object posiitons
    tracker.add_position_to_tracks(tracks)
    
    #camera movement estiator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    #Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    #speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    #Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    # Manually assign referee colors and team label
    referee_ids = [25, 33, 762]
    for frame_num, referee_track in enumerate(tracks['referees']):
        for ref_id in referee_ids:
            if ref_id in referee_track:
                tracks['referees'][frame_num][ref_id]['team'] = -1  # or "referee"
                tracks['referees'][frame_num][ref_id]['team_color'] = (0, 255, 255)  # light cyan for visibility

            
     # Assign Ball Aquisition
    player_ball_assigner = PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        referee_ids = set(tracks.get("meta", {}).get("referee_ids", []))
        assigned_player = player_ball_assigner.assign_ball_to_player(player_track, ball_bbox,referee_ids=referee_ids)

        if assigned_player is not None:
            assigned_player = int(assigned_player)
            if assigned_player in tracks['players'][frame_num]:  # ✅ check key exists
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
            else:
                print(f"⚠️ assigned_player {assigned_player} not found in frame {frame_num}")
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)
    team_ball_control = np.array(team_ball_control)

    
    #save cropped iage of a player
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]
        
        #crop bbox from frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
        #save the cropped image
        cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
        break
    
    # Draw ouptut
    #draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    #draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    
    #draw speed and distance
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    #save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()