#!/usr/bin/env python

from __future__ import print_function
from abc import ABCMeta, abstractmethod


class InterfaceEnvironment:
    """Абстрактный базовый класс окружающей среды

    Примечание:
        abstractmethod являются обязательными методами.
    """

    __metaclass__ = ABCMeta

    def __init__(self, env_params):
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)

    @abstractmethod
    def init(self, env_info={}):
        """Настройка для среды, вызываемой при первом запуске эксперимента.

        Примечание:
            Инициализируйте кортеж с наградой, первым наблюдением состояния, указателем на завершение.
        """

    @abstractmethod
    def start(self):
        """Запускается для приведеня среды в первое состояние до запуска агента.

        Возвращается:
            Первое наблюдение состояния из окружающей среды.
        """

    @abstractmethod
    def next_step(self, action):
        """Изменение окружающей средой после действия агента.

        Аргументы:
            action: Действие, предпринятое агентом

        Возвращается:
            (float, state, Boolean): кортеж из награды, наблюдения за состоянием и указатель на завершение.
        """

    @abstractmethod
    def cleanup(self):
        """Очистка окружающей среды после завершения контекста"""

    @abstractmethod
    def request(self, data):
        """Данные полученные для изменения окружающей среды или сбора статистики

        Аргументы:
            data: данные, переданные в окружающую среду

        Возвращается:
            ответ
        """