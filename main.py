
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from pettingzoo.butterfly import cooperative_pong_v5

def train():

    env = cooperative_pong_v5.parallel_env(
        ball_speed=6,
        left_paddle_speed=12,
        right_paddle_speed=12,
        cake_paddle=True,
        max_cycles=1000,
        bounce_randomness=False,
        max_reward=100,
        off_screen_penalty=-10,
        render_ratio=2,
        kernel_window_length=2,
        render_mode="human",
    )

    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")
    model = PPO(
        CnnPolicy,
        env,
        verbose=3,
        gamma=0.95,
        n_steps=512,
        ent_coef=0.0905168,
        learning_rate=0.00012211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=10,
        clip_range=0.3,
        batch_size=512,
    )
    model.learn(total_timesteps=200000)
    model.save("policy")

    env = cooperative_pong_v5.env()
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)

    model = PPO.load("policy")

    env.reset()
    state = env.reset()[0]
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        act = model.predict(obs, deterministic=True)[0] if not done else None
        env.step(act)
        env.render()

def predict():

   env = cooperative_pong_v5.env(
      ball_speed = 2
   )
   env = ss.color_reduction_v0(env, mode="B")
   env = ss.resize_v1(env, x_size=84, y_size=84)
   env = ss.frame_stack_v1(env, 3)

   model = PPO.load("policy")
   env.reset()
   for agent in env.agent_iter():
      obs, reward, done, info = env.last()
      act = model.predict(obs, deterministic=True)[0] if not done else None
      env.step(act)
      env.render()      

if __name__ == "__main__":
	train()
    #predict()