import math
import numpy as np
import genesis as gs
import torch
import numpy as np
from rsl_rl.env import VecEnv
from genesis.utils.geom import quat_to_xyz,xyz_to_quat,quat_to_R, transform_by_quat, inv_quat, transform_quat_by_quat
import torch.nn.functional as F


def generate_random_positions(base_pos, x_range, y_range, n_envs, device):
    random_x = torch.rand(n_envs, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    random_y = torch.rand(n_envs, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    z_pos = torch.full((n_envs,), base_pos[2], device=device)
    ball_positions = torch.stack([random_x, random_y, z_pos], dim=1)
    
    
    return ball_positions

class PandaBallHitEnv(VecEnv):
    Q_MIN = torch.tensor([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-0.01,-0.01])
    Q_MAX = torch.tensor([ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01])

    JOINT_ANGLE_MIN = torch.tensor([-3.0,-3.0, -3.0, -3.0,-3.0, -3.0, -3.0, 0.0,  0.0])
    JOINT_ANGLE_MAX = torch.tensor([ 3.0, 3.0,  3.0,  3.0, 3.0,  3.0,  3.0, 0.04, 0.04])
    GRIPPER_MIN = 0.0
    GRIPPER_MAX = 0.08
    def __init__(self, cfg: dict | object,num_envs=1, visible=False):
        self.device = gs.device 

        self.cfg = cfg
        self.num_envs = num_envs
        self.num_actions = 9
        self.num_obs = 30 
        self.dt = 0.01  
        self.episode_length_s=2 
        self.max_episode_length = math.ceil(self.episode_length_s / self.dt)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device)

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3, -2, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=self.dt,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                box_box_detection=True,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=visible,
        )
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml"),
        )
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=0.03,  # ボールの半径
                pos=(0.65, 0.0, 0.03), # 初期位置
            ),
            visualize_contact=True,
        )
        self.pole = self.scene.add_entity(
            gs.morphs.Cylinder(
                radius=0.02, # ポールの半径
                height=0.2, # ポールの高さ
                pos=(1.5, 0.0, 0.1), # ポールの初期位置 (ボールから少し離れた場所)
            ),
            visualize_contact=True,
        )



        self.Q_MIN = self.Q_MIN.to(self.device)
        self.Q_MAX = self.Q_MAX.to(self.device)
        self.JOINT_ANGLE_MIN = self.JOINT_ANGLE_MIN.to(self.device)
        self.JOINT_ANGLE_MAX = self.JOINT_ANGLE_MAX.to(self.device)

        self.scene.build(n_envs=self.num_envs, env_spacing=(3.0, 3.0))
        self.envs_idx = np.arange(self.num_envs)
        self._initialize_robot_state()

        self.robot_current_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_current_quat = torch.zeros((self.num_envs, 4), device=self.device)
    

        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.ball_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ball_start_pos=torch.zeros((self.num_envs, 3), device=self.device)
        self.ball_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.pole_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.pole_quat = torch.zeros((self.num_envs, 4), device=self.device)        
        self.place_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.place_pos_test=torch.tensor([0.5, 0.0, 0.4], device=self.device).repeat(self.num_envs, 1)
        self.place_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.episode_num=0

        self.tube_pos = torch.tensor([0.65, 0.0, 0.02], device=self.device).repeat(self.num_envs, 1)
        self.tube_quat = torch.tensor([0.0, 0.0, 0.0,1.0], device=self.device).repeat(self.num_envs, 1)

        

    def _initialize_robot_state(self):
        self.place_pos = torch.tensor([0.4, 0.0, 0.4], device=self.device)

        self.robot_all_dof= torch.arange(9).to(self.device)
        self.motors_dof = torch.arange(7).to(self.device)
        self.fingers_dof = torch.arange(7, 9).to(self.device)
        robot_pos = torch.tensor([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]).to(self.device)
        robot_pos = robot_pos.unsqueeze(0).repeat(self.num_envs, 1) 
        self.robot.set_qpos(robot_pos, envs_idx=self.envs_idx)
        self.scene.step()
    def reset_selected_environments(self, envs_idx):
        if len(envs_idx) == 0:
            return

        robot_joint_pos = torch.tensor([-1.1662,  1.2605, 1.7528, -1.8180, -1.2245,  1.4785,  1.4820,  0.0400,0.0400],dtype=torch.float32).to(self.device)
        self.dof_pos[envs_idx] = robot_joint_pos 
        
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx, :7], 
            dofs_idx_local=self.motors_dof,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.set_dofs_position(position=self.dof_pos[envs_idx, 7:9],zero_velocity=True, dofs_idx_local=self.fingers_dof, envs_idx=envs_idx)

        base_pos = torch.tensor([0.65, 0.0, 0.02], device=self.device)
        x_range = torch.tensor([0.64-self.episode_num, 0.66+self.episode_num], device=self.device) 
        y_range = torch.tensor([-0.01-self.episode_num, 0.01+self.episode_num], device=self.device)
        #y_range = torch.tensor([-0.25-self.episode_num, 0.25+self.episode_num], device=self.device)

        #ランダム
        ball_pos = generate_random_positions(base_pos, x_range, y_range, len(envs_idx), self.device)
        self.ball_pos[envs_idx] = ball_pos
        self.ball_start_pos=self.ball_pos
        #self.ball_quat[envs_idx] = quaternions
        
        """
        #固定
        fixed_pos = torch.tensor([0.65, 0.0, 0.02], device=self.device)
        self.ball_pos[envs_idx] = fixed_pos.unsqueeze(0).repeat(len(envs_idx), 1)
        """
        fixed_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # 無回転（w, x, y, z）
        self.ball_quat[envs_idx] = fixed_quat.unsqueeze(0).repeat(len(envs_idx), 1)

        self.ball.set_pos(self.ball_pos[envs_idx], envs_idx=envs_idx)
        self.ball.set_quat(self.ball_quat[envs_idx], envs_idx=envs_idx)

        fixed_pos = torch.tensor([1.5, 0.0, 0.1], device=self.device)
        fixed_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # 無回転（w, x, y, z）
        self.pole_pos[envs_idx] = fixed_pos.unsqueeze(0).repeat(len(envs_idx), 1)
        self.pole_quat[envs_idx] = fixed_quat.unsqueeze(0).repeat(len(envs_idx), 1)

        self.pole.set_pos(self.pole_pos[envs_idx], envs_idx=envs_idx)
        self.pole.set_quat(self.pole_quat[envs_idx], envs_idx=envs_idx)        

        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.episode_num+=0.0001
        if self.episode_num > 0.2:
            self.episode_num=0
            print('reset')


    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_buf[:] = True
        self.reset_selected_environments(torch.arange(self.num_envs, device=gs.device))
        state, info = self.get_observations()
        self.scene.step()
        return state, info


    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions= torch.tanh(actions)

        actions[:, 7:9]= actions[:, 7:9]/100.0
        delta_actions = torch.clamp(actions, min=self.Q_MIN, max=self.Q_MAX)
        current_actions = self.robot.get_dofs_position(self.robot_all_dof, self.envs_idx).clone().detach()
        control_actions = current_actions + delta_actions
        
        control_actions=torch.clamp(control_actions,min=self.JOINT_ANGLE_MIN, max=self.JOINT_ANGLE_MAX)
        self.robot.control_dofs_position(control_actions, self.robot_all_dof, self.envs_idx)
        self.scene.step()

        states, info = self.get_observations()
        rewards = self.compute_rewards(states, info)
        dones   = self._check_termination_conditions(states, info)
        self.robot_current_pos = info["observations"]["gripper_position"]
        self.robot_current_quat = info["observations"]["gripper_quaternion"]
        self.reset_selected_environments(self.reset_buf.nonzero(as_tuple=False).flatten())
        return states, rewards, dones, info
    

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        block_position = self.ball.get_pos()
        block_quaternion = self.ball.get_quat()
        gripper_position = self.robot.get_link("hand").get_pos()
        gripper_quaternion = self.robot.get_link("hand").get_quat()
        all_dof_pos = self.robot.get_dofs_position(self.robot_all_dof, self.envs_idx)
        
        ball_position = self.ball.get_pos()
        pole_position = self.pole.get_pos()
        #print('ball:'+str(ball_position))
        #print('pole:'+str(pole_position))

        gripper2ball_distance = torch.norm(ball_position - gripper_position, dim=1, keepdim=True)
        ball2pole_distance = torch.norm(ball_position - pole_position, dim=1, keepdim=True)

        is_contact_ball = self.get_robot_ball_contacts()
        left_position = self.robot.get_link("left_finger").get_pos()
        right_position = self.robot.get_link("right_finger").get_pos()
        left_distance = torch.norm(left_position - block_position, dim=1, keepdim=True)
        right_distance = torch.norm(right_position - block_position, dim=1, keepdim=True)
        distance_diff = torch.abs(left_distance - right_distance)

        states_dict = {
            "observations": {
                "block_quaternion": block_quaternion,
                "block_position": block_position,
                "gripper_position": gripper_position,
                "gripper_quaternion": gripper_quaternion,
                "robot_all_dof": all_dof_pos,
                "gripper2ball_distance": gripper2ball_distance,
                "ball2pole_distance": ball2pole_distance,
                "place_pos": self.place_pos_test,
                "is_contact_ball": is_contact_ball,
                "distance_diff": distance_diff,
            }
        }
        
        states = torch.cat([
            block_position,
            gripper_position,
            gripper2ball_distance,
            self.place_pos_test,
            ball2pole_distance,
            all_dof_pos,
            gripper_quaternion,
            block_quaternion,
            is_contact_ball,
        ], dim=1)
        
        return states, states_dict
    def compute_rewards(self, states, states_dict):
        obs = states_dict["observations"]
        g2b_dist = obs["gripper2ball_distance"].squeeze(-1) # グリッパーとボールの距離
        b2p_dist = obs["ball2pole_distance"].squeeze(-1)    # ボールとポールの距離
        #ball_vel_norm = torch.norm(obs["ball_velocity"], dim=1) # ボールの速度のノルム
        # 1. グリッパーがボールに近づくほど報酬
        # 距離が近いほど高い報酬を与えるように調整
        approach_reward = 1.0 / (1.0 + g2b_dist * 10.0) # 距離に反比例する報酬

        # 2. ボールがポールに近づくほど報酬
        # ボールがポールに当たった場合に大きな報酬
        hit_reward_threshold = 0.05 # ボールがポールに当たったとみなす距離
        hit_reward = (b2p_dist < hit_reward_threshold).float() * 100000.0 # 当たったら大きな報酬

        # 3. ボールが動いていることへの報酬 (初期接触後)
        # グリッパーがボールに接触している、かつボールが動いている場合に報酬
        contact_with_ball = self.get_robot_ball_contacts().any(dim=1).float()
        #ball_movement_reward = contact_with_ball * ball_vel_norm * 10.0 # 接触中にボールの速度に比例
        ball_movement_reward = contact_with_ball * 10.0 # 接触中にボールの速度に比例
        org_dist=0.85 #0.35 ball ps 1.0  0.85 ball pos 1.5
        b2p_reward= (1.0 / (1.0 + b2p_dist * 10.0)- 1.0 / (1.0 + org_dist * 10.0))*1000 # 距離に反比例する報酬

        # 4. エピソード時間に対するペナルティ
        time_penalty = -0.01 * self.episode_length_buf

        # 総報酬
        #total_reward = approach_reward + hit_reward + ball_movement_reward + time_penalty
        total_reward = approach_reward + hit_reward + b2p_reward + time_penalty
        #total_reward = approach_reward + hit_reward + b2p_reward+ ball_movement_reward + time_penalty
        #print('b2p:'+str(b2p_reward)) 
        #print('b2p:'+str(b2p_dist)) 

        return total_reward

    def get_robot_ball_contacts(self):
        """
        ロボットとボールの接触状態を検出します。
        Returns:
            torch.Tensor: 各環境におけるロボットとボールの接触状態 (ブール値のテンソル)。
        """
        contacts = self.robot.get_contacts(self.ball)
        valid_mask = torch.from_numpy(contacts["valid_mask"]).to(self.device)
        is_contact = valid_mask.any(dim=1).float().unsqueeze(1) # (num_envs, 1)
        return is_contact
    """
    def get_robot_ball_contacts(self):
        LEFT_LINK = 10
        RIGHT_LINK = 11

        contacts = self.robot.get_contacts(self.ball)
        # NumPy → Tensor に変換して self.device に移動
        valid_mask = torch.from_numpy(contacts["valid_mask"]).to(self.device)
        link_a     = torch.from_numpy(contacts["link_a"]    ).to(self.device)

        def _check_contact(link_id):
            # ここも self.device を使う
            link_tensor = torch.tensor([link_id], device=self.device).repeat(self.num_envs, 1)
            isin_a = torch.logical_and(torch.isin(link_a, link_tensor), valid_mask)
            # バッチごとに any
            return isin_a.any(dim=1).float()

        contact_left  = _check_contact(LEFT_LINK)
        contact_right = _check_contact(RIGHT_LINK)
        return torch.stack([contact_left, contact_right], dim=1)
    """

    def get_robot_contacts(self, obj):
        # 1) contacts を取得
        contacts = self.robot.get_contacts(obj)

        # 2) NumPy 配列 → Tensor 変換 & デバイス移動
        valid_mask = torch.from_numpy(contacts["valid_mask"]).to(self.device)
        link_a     = torch.from_numpy(contacts["link_a"]    ).to(self.device)
        link_b     = torch.from_numpy(contacts["link_b"]    ).to(self.device)

        # 3) 与えられたリンクIDが a,b の両方で接触しているかをチェックするヘルパー
        def _check_contact(link_id):
            # device は self.device を使う
            link_tensor = torch.tensor([link_id], device=self.device).repeat(self.num_envs, 1)
            hit_a = torch.logical_and(torch.isin(link_a, link_tensor), valid_mask)
            hit_b = torch.logical_and(torch.isin(link_b, link_tensor), valid_mask)
            # 各 env ごとに a,b 両方とも当たっていれば 1.0、そうでなければ 0.0
            return (hit_a.any(dim=1) & hit_b.any(dim=1)).float()

        # 例として、平面 (plane) のリンク 0,1 をチェックする場合
        contact_link0 = _check_contact(0)
        contact_link1 = _check_contact(1)
        # 必要に応じて他のリンクIDも同様に

        # 複数リンクをまとめて返す場合はスタック
        return torch.stack([contact_link0, contact_link1], dim=1)


    def _check_termination_conditions(self, states, states_dict):
        # step ごとにカウント
        self.episode_length_buf += 1

        # (1) 時間超過フラグ  shape: [num_envs,]
        time_exceeded = self.episode_length_buf > self.max_episode_length

        # (2) 目標達成フラグ  shape: [num_envs,]
        # block2target_distance は (num_envs,1) なので squeeze
        #block2target = states_dict["observations"]["ball2pole_distance"].squeeze(-1)
        #task_complete = block2target < 0.2

        # (3) 平面への接触フラグ  shape: [num_envs,] または誤って (num_envs,2) の場合は any で落とす
        contact_bool = self.get_robot_contacts(self.plane)
        if contact_bool.dim() > 1:
            contact_bool = contact_bool.any(dim=1)
        contact_bool = contact_bool.bool()

        # (4) いずれかが True ならリセット
        #reset = time_exceeded | task_complete | contact_bool  # shape: [num_envs,]
        reset = time_exceeded | contact_bool  # shape: [num_envs,]


        # reset_buf を同じ型で更新して返す
        self.reset_buf = reset.to(self.reset_buf.dtype)
        return reset.clone()


if __name__ == "__main__":
    gs.init(backend=gs.gpu, precision="32",logging_level='warning')
    

    env_cfg = {
        "num_actions": 9,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 2.0,
        "resampling_time_s": 4.0,
        "action_scale": 1.0,
        "simulate_action_latency": True,
        "clip_actions": 1.0,
    }

    env = PandaBallHitEnv(cfg=env_cfg,num_envs=25,  visible=True)
    state, info = env.reset()
    print(state)
    for i in range(1000):
        actions = 2*torch.rand((env.num_envs, 9), device=env.device) - 1
        states, rewards, dones, info = env.step(actions)
        # print(rewards)
    print(rewards)
    print(states)

