import numpy as np

def calculate_player_metrics(tracks, target_player_id):
    player_metrics = {}
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            if player_id == target_player_id:
                if player_id not in player_metrics:
                    player_metrics[player_id] = {
                        'matches': 0, 'goals': 0, 'assists': 0,
                        'fitness_score': np.random.randint(70, 100),
                        'win_rate': 0, 'performance_score': 0
                    }

                
                if track.get('has_ball', False):
                    if np.random.rand() < 0.05:  
                        player_metrics[player_id]['goals'] += 1
                    if np.random.rand() < 0.1:  
                        player_metrics[player_id]['assists'] += 1
                
                player_metrics[player_id]['matches'] += 1
    
    if target_player_id in player_metrics:
        player_metrics[target_player_id]['win_rate'] = np.random.randint(50, 100)
        player_metrics[target_player_id]['performance_score'] = round(np.random.uniform(6, 10), 1)
        return player_metrics[target_player_id]
    else:
        return f"Player ID {target_player_id} not found in the given tracks."