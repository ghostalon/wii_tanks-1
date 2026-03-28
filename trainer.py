import os
import pygame
import torch
from Constants import *
from Enviorment import Enviroment
from Human_Agent import Human_Agent
from Dqn_Agent import DQN_Agent
from Random_Agent import Random_Agent
from ReplayBuffer import ReplayBuffer
from Advanced_Random_Agent import Advanced_Random_Agent
import wandb

def main():
    num = 4
    env = Enviroment()
    env.init_screen()
    buffer = ReplayBuffer(path=None)
    
    player1 = DQN_Agent(device=torch.device('cpu'))
    player1_hat = DQN_Agent(device=torch.device('cpu'))
    # player2 = Human_Agent()
    player2 = Random_Agent()
    player2 = Advanced_Random_Agent(OPPONENT_EPSILON)

    pygame.init()

    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    backround = pygame.image.load("Data/wood.png")
    screen.blit(backround, (0,0))
    pygame.display.set_caption("TANKS!")

    clock = pygame.time.Clock()
    FPS = 60

    run = True
    eplosion_timer = 0
    game_ended = False
    result = 0
    learning_rate = LEARNING_RATE
    start_epoch = 0
    C  = TARGET_UPDATE_FREQ
    loss = torch.tensor(0)
    avg = 0
    scores, losses, avg_score = [], [], []
    optim = torch.optim.Adam(player1.DQN.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,100000, gamma=0.50)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[5000*1000, 10000*1000, 15000*1000, 20000*1000, 25000*1000, 30000*1000], gamma=0.5)
    step = 0

    win_count = 0
    win_rate_window = []   # 1=win, 0=lose for last 100 episodes
    steps_to_win_list = []
    steps_to_lose_list = []
    grad_steps = 0  # total gradient updates across all episodes

    project = "TankAI"
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        name=f'{project}_{num}',
        id=f'{project}_{num}',
        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "Schedule": f'{str(scheduler.milestones)} gamma={str(scheduler.gamma)}',
        "epochs": epochs,
        "start_epoch": start_epoch,
        "epsilon_start": epsilon_start,
        "epsilon_final": epsilon_final,
        "decay": epsiln_decay,
        "gamma": gamma,
        "batch_size": batch_size, 
        "MIN_BUFFER": MIN_BUFFER,
        "C": C,
        "WIN_RATE_WINDOW": WIN_RATE_WINDOW,
        "CHECKPOINT_INTERVAL": CHECKPOINT_INTERVAL,
        "OPPONENT_EPSILON": OPPONENT_EPSILON,
        "Model":str(player1.DQN),
        "TANK_SPEED": TANK_SPEED,
        "BULLET_SPEED": BULLET_SPEED,
        "MAX_AMMUNITION": MAX_AMMUNITION,
        "REWARD_WIN": REWARD_WIN,
        "REWARD_LOSE": REWARD_LOSE,
        "STEP_PENALTY": STEP_PENALTY,
        "K_AIM": K_AIM,
        "AIM_BONUS_NEAR": AIM_BONUS_NEAR,
        "AIM_THRESH_NEAR": AIM_THRESH_NEAR,
        "AIM_BONUS_LOCK": AIM_BONUS_LOCK,
        "AIM_THRESH_LOCK": AIM_THRESH_LOCK,
        "SHOOT_PENALTY": SHOOT_PENALTY,
        "SHOOT_BONUS_AIMED": SHOOT_BONUS_AIMED,
        "K_DANGER": K_DANGER,
        })
    
    
    for epoch in range(epochs):
        print(epoch, end='\r')
        env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        loss_steps = 0
        step = 0
        while True:
            print(step, end='\r')
            step += 1
            pygame.event.pump()
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return
                    
            state = env.state()
            action1 = player1.get_Action(events=events, state = state, epoch=epoch)
            action2 = player2.get_Action(events=events, state = env.state(), epoch=epoch)

            env.move(action1, action2)
            env.render()
            next = env.state()
            done = env.end_of_game()
            reward = env.reward(state, action1, next, done)
            episode_reward += reward
            buffer.push(state, action1, reward, next, done)


            if done != 0:
                # Episode finished: record metrics and optionally checkpoint
                avg_loss = (episode_loss / loss_steps) if loss_steps > 0 else 0.0
                scores.append(episode_reward)
                losses.append(avg_loss)
                if done == 1:
                    win_count += 1
                    steps_to_win_list.append(step)
                    win_rate_window.append(1)
                else:
                    steps_to_lose_list.append(step)
                    win_rate_window.append(0)
                if len(win_rate_window) > WIN_RATE_WINDOW:
                    win_rate_window.pop(0)
                win_rate = sum(win_rate_window) / len(win_rate_window)
                print(f"num {num} Epoch {epoch} step {step} Reward {episode_reward:.3f}  AvgLoss {avg_loss:.4f}  Wins {win_count}  WinRate {win_rate:.2f}")
                wandb.log({
                    "episode_reward": episode_reward,
                    "avg_loss": avg_loss,
                    "win_rate_20": win_rate,
                    "steps_this_episode": step,
                    "steps_to_win": steps_to_win_list[-1] if done == 1 else None,
                    "steps_to_lose": steps_to_lose_list[-1] if done == 2 else None,
                })
                if epoch % CHECKPOINT_INTERVAL == 0:
                    os.makedirs('checkpoints', exist_ok=True)
                    player1.save_param(f'checkpoints/dqn_epoch_{epoch}.pth')
                # Restart the game immediately
                break

            state = next

            if len(buffer) < MIN_BUFFER:
                continue

            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            actions_index = player1.actions_to_indices(actions)
            Q_values = player1.Q(states, actions_index)
            # DDQN: online network selects best action, target network evaluates it
            next_actions, _ = player1.get_Actions_Values(next_states)       # online picks action
            Q_hat_values = player1_hat.Q(next_states, next_actions)         # target evaluates it
            loss = player1.DQN.loss(Q_values, rewards, Q_hat_values, dones)
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            grad_steps += 1
            try:
                lv = float(loss.item())
            except Exception:
                lv = 0.0
            episode_loss += lv
            loss_steps += 1

            if grad_steps % C == 0:
                player1_hat.fix_update(dqn=player1.DQN)
            
            pygame.display.update()
            clock.tick(FPS)
            

    
if __name__ == "__main__":
    main()