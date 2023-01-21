import random
import time

import gym
import pygame

pygame.init()
pygame.display.init()
pygame.font.init()

class Arm:
    def __init__(self, p=0, earn=0):
        self.probability = p
        self.earn = earn

    def pull(self):
        return self.earn if random.random() < self.probability else 0

    def __str__(self):
        return f"Arm: p: {self.probability}, earn: {self.earn}"

    def __repr__(self):
        return self.__str__()

class TwoArmedBanditEnv(gym.Env):
    MACHINE = pygame.image.load("./assets/graphics/slot-machine.png")
    ARROW = pygame.image.load("./assets/graphics/up_arrow.png")
    FONT = pygame.font.Font("./assets/fonts/font.ttf", 64)
    
    MACHINE_WIDTH, MACHINE_HEIGHT = MACHINE.get_size()

    WINDOWS_WIDTH = MACHINE_WIDTH * 2 + 150
    WINDOWS_HEIGHT = MACHINE_HEIGHT + 200

    def __init__(self):
        self.arms = (
            Arm(0.5, 1),
            Arm(0.1, 100)
        )
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(len(self.arms))
        self.action = None
        self.reward = None
        self.total_reward = 0

        self.window = pygame.display.set_mode((self.WINDOWS_WIDTH, self.WINDOWS_HEIGHT))
        pygame.display.set_caption("Two-Armed Bandit Environment")

    def _get_observations(self):
        return 0

    def _get_info(self):
        return { 'state': 0 }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_observations(), self._get_info()

    def step(self, action):
        self.action = action
        self.reward = self.arms[action].pull()
        self.render()
        self.total_reward += self.reward
        return self._get_observations(), self.reward, False, False, self._get_info()

    def render(self):
        self.window.fill((0, 0, 0))

        # render first machine
        self.window.blit(self.MACHINE, (50, 100))

        # render second machine
        self.window.blit(self.MACHINE, (100 + self.MACHINE_WIDTH, 100))
        
        # render the total reward
        text_obj = self.FONT.render(f"${self.total_reward}", True, (133, 187, 101))
        text_rect = text_obj.get_rect()
        text_rect.center = (self.WINDOWS_WIDTH * 0.5 , 50)
        self.window.blit(text_obj, text_rect)

        pygame.display.update()

        time.sleep(0.5)

        # render the arrow under the selected machine
        x = 50 + self.MACHINE_WIDTH * 0.5
        
        if (self.action == 1):
            x += self.MACHINE_WIDTH + 50

        w, h = self.ARROW.get_size()
        y = self.WINDOWS_HEIGHT - 50 - h * 0.5

        self.window.blit(self.ARROW, (x - w * 0.5 - 80, y))
        
        # render the reward
        text_obj = self.FONT.render(f"+{self.reward}", True, (255, 250, 26))
        text_rect = text_obj.get_rect()
        text_rect.center = (x, 50)
        self.window.blit(text_obj, text_rect)
        
        pygame.event.pump()
        pygame.display.update()

        time.sleep(0.5)

    def close(self):
        pygame.display.quit()
        pygame.font.quit()
        pygame.quit()