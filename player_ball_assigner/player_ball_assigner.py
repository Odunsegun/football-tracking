import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
        self.last_assigned_player_id = None
        self.candidate_player_id = None
        self.candidate_frame_count = 0
        self.frames_required_to_switch = 3
    
    def assign_ball_to_player(self,players,ball_bbox,referee_ids=None):
        ball_position = get_center_of_bbox(ball_bbox)
        referee_ids = referee_ids or set()
        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            if player_id in referee_ids:  # ✅ Move this check out of the nested loop
                continue
            
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id
                    
        # If no player is close, reset everything
        if assigned_player == -1:
            self.candidate_player_id = None
            self.candidate_frame_count = 0
            return self.last_assigned_player_id  # keep last owner

        # If same as last assigned, no need to confirm again
        if assigned_player == self.last_assigned_player_id:
            self.candidate_player_id = None
            self.candidate_frame_count = 0
            return self.last_assigned_player_id

        # If same as candidate, increment frame count
        if assigned_player == self.candidate_player_id:
            self.candidate_frame_count += 1
        else:
            self.candidate_player_id = assigned_player
            self.candidate_frame_count = 1

        # Confirm switch if held for enough frames
        if self.candidate_frame_count >= self.frames_required_to_switch:
            self.last_assigned_player_id = self.candidate_player_id
            self.candidate_player_id = None
            self.candidate_frame_count = 0

        return self.last_assigned_player_id


        #return assigned_player