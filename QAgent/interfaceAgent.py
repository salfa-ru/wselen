#!/usr/bin/env python

from __future__ import print_function
from abc import ABCMeta, abstractmethod


class InterfaceAgent:
    """Абстрактный класс агента 
    Примечание:
        abstractmethod являются обязательными методами.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def init(self, agent_info):
        """Настройка для агента перед первым запуском."""

    @abstractmethod
    def start(self, observation):
        """Первый метод вызывается после запуска среды.
        Аргументы:
            observation (Numpy array): наблюдение за состоянием внешней среды.
        Возвращается:
            Первое действие, которое предпринимает агент.
        """

    @abstractmethod
    def next_step(self, reward, observation):
        """Шаг агента.
        Аргументы:
            reward (float): награда, полученная за последнее действия
            observation (Numpy array): наблюдение за состоянием внешней среды
        Возвращается:
            Действие, которое предпринимает агент.
        """

    @abstractmethod
    def end(self, reward):
        """Завершение работы агента.
        Аргументы:
            reward (float): вознаграждение, полученное агентом во время завершения.
        """

    @abstractmethod
    def cleanup(self):
        """Очистка окружающей среды после завершения контекста."""

    @abstractmethod
    def request(self, data):
        """Данные полученные для изменения агента или сбора статистики
        Аргументы:
            data: данные, переданные агенту.
        Возвращается:
            ответ.
        """