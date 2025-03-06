from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_analysis import PlayerBallAssigner
from view_transformer import ViewTransformer
from speed_distance_estimator import SpeedAndDistance_Estimator
from utils import read_video, save_video, get_center_of_bbox, get_bbox_width, measure_distance, measure_xy_distance, get_foot_position
from trackers import Tracker
# from player_metrics import calculate_player_metrics
import os
from moviepy import VideoFileClip

def convert_to_h264(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

def main(file_name):
    video_frames=read_video(file_name)
    tracker=Tracker('final_best.pt')
    tracks=tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path=r'stubs\track_stubs.pkl')
    tracker.add_position_to_tracks(tracks)
    ## Camera movement estimator

    # camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    # camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
    #                                                                             read_from_stub=True,
    #                                                                             stub_path='stubs\\camera_movement_stub.pkl')
    # camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    ## View transformer

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    ## Interpolate Ball positions

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    ## Speed and Distance

    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    ## Assignment of player teams

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


    ## Assign Ball

    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    ## Draw object tracks


    output_video_frames=tracker.draw_annotations(video_frames,tracks,team_ball_control)

    ## Draw Camera movement
    # output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

     ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    # output_frames_filepath = r'static\output\output_frames.npy' # Define the file path
    # np.save(output_frames_filepath, np.array(output_video_frames))

    save_video(output_video_frames,'static/output/output_video.avi')
    convert_to_h264('static/output/output_video.avi','static/output/output_video.mp4')
    os.remove('static/output/output_video.avi')
    # final_frame = len(video_frames) - 1
    # player_metrics = calculate_player_metrics({'players': [tracks['players'][final_frame]]})  # Only final frame
    # # return player_metrics
    # print(player_metrics)
# main("videoplayback (online-video-cutter.com).mp4")