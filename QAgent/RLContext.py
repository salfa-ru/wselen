#!/usr/bin/env python

from __future__ import print_function
import seaborn as sns
sns.set_style('darkgrid')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output
import time
 

def createField(map_dim, obstacles, start_state, end_state, current_state): 
    fig, ax = plt.subplots(1, figsize=(map_dim[0], map_dim[1])) 
    rect = patches.Rectangle((0, map_dim[1]), map_dim[0], -map_dim[1], linewidth = 2, edgecolor = 'black') 
    ax.add_patch(rect)
    
    for i in range(map_dim[0]):
        for j in range(map_dim[1]):
            rect = patches.Rectangle((i,j+1), 1, -1, linewidth = 1, edgecolor = 'black', facecolor = 'white', alpha = 0.05) 
            ax.add_patch(rect)

    rect = patches.Rectangle((start_state[0],start_state[1]+1), 1, -1, linewidth = 1, edgecolor = 'black', facecolor = 'white', alpha = 0.3) 
    ax.add_patch(rect)         
    for state in obstacles:
        rect = patches.Rectangle((state[0],state[1]+1), 1, -1, linewidth = 1, edgecolor = 'black', facecolor = 'red', alpha = 0.5)
        ax.add_patch(rect)
  
    rect = patches.Rectangle((end_state[0],end_state[1]+1), 1, -1, linewidth = 1, edgecolor = 'black', facecolor = 'green', alpha = 0.4) 
  
    ax.add_patch(rect) 
  
    plt.xticks(np.arange(map_dim[0]+1))
    plt.yticks(np.arange(map_dim[1]+1))

    return fig,ax


def print_path_agent(a, f, current_state, map_dim, steps_agent, model, q_values):
    l = len(steps_agent)
    for idx, p in enumerate(steps_agent):
        x1, x2 = int(p[0] / 10), int(p[1] / 10)
        y1, y2 = p[0] - x1*10, p[1] - x2*10
        x = [x1 + 0.5, x2 + 0.5]
        y = [y1 + 0.5, y2 + 0.5]
        a.plot(x, y, marker = 'o', color = 'y', linewidth = 2, alpha = (idx+1)/ l)
    
    x = [current_state[0] + 0.5, current_state[0] + 0.5]
    y = [current_state[1] + 0.5, current_state[1] + 0.5]
    a.plot(x, y, marker = 'o', color = 'r', linewidth = 4)
            
    for val in model.keys():
        x = int(val/10)
        y = val - x * 10
        
        txt = f'x:{x} y:{y}\n'
        lst = list(model[val].keys())
        lst.sort()
        for k in lst:
            q = round(q_values[val][k], 6) if val in q_values.keys() else 0.0
            txt += f'{k} {model[val][k][1]}-{q}\n'
        
        a.text(x + 0.05, y - 0.05, txt, fontsize = 8, alpha = 0.7)
    


class RLContext:
    """Объединяет состояния агента и состояние наблюдаемой окружающей среды.
    Аргументы:
        env_name (string): имя модуля, в котором можно найти класс среды
        agent_name (string): имя модуля, в котором можно найти класс агента
    """

    def __init__(self, env_class, env_parmas, agent_class, show_map=True):
        self.environment = env_class(env_parmas)
        self.agent = agent_class()
        self.idx_image = 0
        self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None
        self.show_map = show_map
        self.is_terminal = True

    def init(self, agent_init_info={}, env_init_info={}):
        """Начальный метод, вызываемый при создания контекст"""
        self.environment.init(env_init_info)
        self.agent.init(agent_init_info)

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0
        self.is_terminal = False

    def start(self, agent_start_info={}, env_start_info={}):
        """Начинается взаимодействия контекста

        Возвращается:
            tuple: (состояние, действие)
        """

        last_state = self.environment.start()
        self.last_action = self.agent.start(last_state)
        self.agent.num_episodes = self.num_episodes
        
        observation = (last_state, self.last_action)
        self.plot_map_states()
        return observation
    
    def plot_map_states(self):
        """Отображение внешней среды и агента
        """
        if self.show_map:
            clear_output(wait=True) # Очищаем экран

            f,a = createField(self.environment.map_dim,
                        self.environment.obstacles,
                        self.environment.start_state,
                        self.environment.end_state,
                        self.environment.current_state)

            print_path_agent(a, f, self.environment.current_state, self.environment.map_dim, self.agent.steps_agent, self.agent.model, self.agent.q_values)
            plt.title(f" Планирование на {self.agent.planning_steps} шагов №{self.num_episodes}, шаг {self.num_steps} получено очков {self.total_reward}:", fontsize = 16)
            plt.savefig(f'img_{self.idx_image}_{self.agent.planning_steps}_{self.num_episodes}_{self.num_steps}_{self.total_reward}.png', bbox_inches='tight')
            self.idx_image += 1
            plt.show()
            plt.close(f)
            plt.clf()

    def agent_start(self, observation):
        """Запускает агента.

        Аргументы:
            observation: Первое наблюдение из окружающей среды (observation - наблюдение)

        Возвращается:
            Действие, предпринятое агентом.
        """
        return self.agent.start(observation)

    def agent_step(self, reward, observation):
        """Шаг, предпринятый агентом

        Аргументы:
            reward (float): последнее вознаграждение, полученное агентом за совершение последнего действия. (reward - вознаграждение)
            observation : наблюдение за состоянием, которое агент получает из окружающей среды. (observation - наблюдение)

        Возвращается:
            Действие, предпринятое агентом.
        """
        return self.agent.next_step(reward, observation)

    def agent_end(self, reward):
        """Запуск при завершении работы агента

        Аргументы:
            reward (float): вознаграждение, полученное агентом при завершении контекста
        """
        self.agent.end(reward)

    def env_start(self):
        """Запуск наблюденя за внешней средой

        Возвращается:
            (float, state, Boolean): вознаграждение, наблюдение за состоянием, указатель на завершение контекста.
        """
        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.start()
        return this_observation

    def env_step(self, action):
        """Изменения предпринятые средой на основе действия агента

        Аргументы:
            action: Действие, предпринятое агентом.

        Возвращается:
            (float, state, Boolean): вознаграждение, наблюдение за состоянием, указатель на завершение контекста.
        """
        ro = self.environment.next_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro

    def step(self):
        """Шаг, выполняемый контекстом, принимает изменение среды и шаг агента, а также завершение контекста.

        Возвращается:
            (float, state, action, Boolean): награда, последнее наблюдение за состоянием, последнее действие, указатель на завершение контекста.
        """

        (reward, last_state, term) = self.environment.next_step(self.last_action)

        self.total_reward += reward

        if term:
            self.num_episodes += 1
            self.agent.end(reward)
            
            roat = (reward, last_state, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.next_step(reward, last_state)
            roat = (reward, last_state, self.last_action, term)
        
        self.plot_map_states()

        return roat

    def cleanup(self):
        """Очистка параметров контекста, среды и агента."""
        self.environment.cleanup()
        self.agent.cleanup()

    def agent_send_data(self, message):
        """Передача данных для агента

        Аргументы:
            message: данные переданные агенту 

        Возвращается:
            Ответное от агента
        """
        return self.agent.response(message)

    def env_send_data(self, message):
        """Передача данных для окружающей средой

        Аргументы:
            message: данные переданные агенту окружающей среде

        Возвращается:
            Ответное от агента внешней среды 
        """
        return self.environment.response(message)

    def episode(self, max_steps_this_episode):
        """Запуск эпизода контекста

        Аргументы:
            max_steps_this_episode (Int): максимальное количество шагов для выполнения эксперимента в эпизоде

        Возвращается:
            Boolean: если эпизод должен прекратиться
        """
        self.is_terminal = False

        self.num_episodes += 1
        self.num_steps = 0
        self.start()

        while (not self.is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            reward, _, action, is_terminal = self.step()
            self.is_terminal = is_terminal

        return self.is_terminal

    def statistics(self):
        """Возвращает статистику и основные параметры RL контекста

        Возвращается:
            float: Общая награда
        """
        return self.total_reward
