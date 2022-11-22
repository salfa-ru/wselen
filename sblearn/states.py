# -*- coding: utf-8 -*-

from action_library import *
#from action_library import GiveBirth
import random
#from entities import Creature
import action_library
from action_library import *
from substances import *
import collections
import numpy as np

class Action(object):
    def __init__(self, subject):
        self.subject = subject
        self.accomplished = False
        self._done = False
        self.instant = False

    def get_objective(self):
        return {}

    def set_objective(self, control=False, **kwargs):
        valid_objectives = list(self.get_objective().keys())

        for key in list(kwargs.keys()):
            if key not in valid_objectives:
                if control:
                    raise ValueError("{0} is not a valid objective".format(key))
                else:
                    pass  # maybe need to print
            else:
                setattr(self, "_{0}".format(key), kwargs[key])

    def action_possible(self):
        return True

    def do(self):
        self.check_set_results()
        self._done = True

    def check_set_results(self):
        self.accomplished = True

    @property
    def results(self):
        out = {"done": self._done, "accomplished": self.accomplished}
        return out

    def do_results(self):
        self.do()
        return self.results

class GiveBirth(Action):
    def __init__(self, subject, pregnant_state):
        super(GiveBirth, self).__init__(subject)

        self.pregnant_state = pregnant_state

    def action_possible(self):
        cells_around = self.get_empty_cells_around()

        if not cells_around:
            return False

        return True

    def do(self):
        if self.results["done"]:
            return

        if not self.action_possible():
            return

        cells_around = self.get_empty_cells_around()

        place = random.choice(cells_around)

        offspring = Creature()

        self.subject.board.insert_object(place[0], place[1], offspring, epoch_shift=1)

        self.subject.remove_state(self.pregnant_state)

        self._done = True

        self.check_set_results()

    def get_empty_cells_around(self):
        cells_near = []

        if self.subject.board.cell_passable(self.subject.x, self.subject.y + 1):
            cells_near.append((self.subject.x, self.subject.y + 1))
        if self.subject.board.cell_passable(self.subject.x, self.subject.y - 1):
            cells_near.append((self.subject.x, self.subject.y - 1))
        if self.subject.board.cell_passable(self.subject.x + 1, self.subject.y):
            cells_near.append((self.subject.x + 1, self.subject.y))
        if self.subject.board.cell_passable(self.subject.x - 1, self.subject.y):
            cells_near.append((self.subject.x - 1, self.subject.y))

        return cells_near



class State(object):
    def __init__(self, subject):
        self.subject = subject
        self.duration = 0

    def affect(self):
        self.duration += 1


class Pregnant(State):
    def __init__(self, subject):
        super(Pregnant, self).__init__(subject)

        self.timing = 15

    def affect(self):
        super(Pregnant, self).affect()

        if self.duration == self.timing:
            self.subject.action_queue.insert(0, GiveBirth(self.subject, self))


class NotTheRightMood(State):
    def __init__(self, subject):
        super(NotTheRightMood, self).__init__(subject)

        self.timing = 10

    def affect(self):
        super(NotTheRightMood, self).affect()

        if self.duration == self.timing:
            self.subject.remove_state(self)
class LearningMemory(object):
    def __init__(self, host):
        self.host = host
        self.memories = {}

    def save_state(self, state, action):
        self.memories[action] = {"state": state}

    def save_results(self, results, action):
        if action in self.memories:
            self.memories[action]["results"] = results
        else:
            pass

    def make_table(self, action_type):
        table_list = []
        for memory in self.memories:
            if isinstance(memory, action_type):
                if "state" not in self.memories[memory] or "results" not in self.memories[memory]:
                    continue
                row = self.memories[memory]["state"][:]
                row.append(self.memories[memory]["results"])
                table_list.append(row)

        return table_list

    def obliviate(self):
        self.memories = {}

class Entity(object):
    def __init__(self):
        # home universe
        self.board = None

        # time-space coordinates
        self.x = None
        self.y = None
        self.z = None

        # lifecycle properties
        self.age = 0
        self.alive = False
        self.time_of_death = None

        # action queues
        self.action_queue = []
        self.action_log = []

        # common properties
        self.passable = False
        self.scenery = True
        self._container = []
        self._states_list = []

        # visualization properties
        self.color = "#004400"

    def __str__(self):
        raise Exception

    @classmethod
    def class_name(cls):
        return "Entity"

    def live(self):
        self.get_affected()
        self.z += 1
        self.age += 1

    def get_affected(self):
        for state in self._states_list:
            state.affect()

    def has_state(self, state_type):
        for state in self._states_list:
            if isinstance(state, state_type):
                return True
        return False

    def add_state(self, state):
        self._states_list.append(state)

    def remove_state(self, state):
        self._states_list.remove(state)

    def contains(self, substance_type):
        for element in self._container:
            if type(element) == substance_type:
                return True
        return False

    def extract(self, substance_type):
        substance_index = None
        for i, element in enumerate(self._container):
            if type(element) == substance_type:
                substance_index = i
                break
        if substance_index is None:
            return None
        return self._container.pop(substance_index)

    def pocket(self, substance_object):
        if substance_object is not None:
            self._container.append(substance_object)

    def dissolve(self):
        self.board.remove_object(self)

    def count_substance_of_type(self, type_of_substance):
        num = 0
        for element in self._container:
            if isinstance(element, type_of_substance):
                num += 1

        return num
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        self.passable = False
        self.scenery = False
        self.name = ''
        self.sex = random.choice([True, False])

        self.private_learning_memory = LearningMemory(self)
        self.public_memory = None

        self.private_decision_model = None
        self.public_decision_model = None

        self.plan_callable = None

        self.memory_type = ""
        self.model_type = ""

        self.memory_batch_size = 1

        self.memorize_tasks = {}
        self.chosen_action = None

    @classmethod
    def class_name(cls):
        return "Agent"

    def pre_actions(self, refuse):
        return True

    def live(self):
        super(Agent, self).live()

        if not self.pre_actions():
            return

        if self.need_to_update_plan():
            self.plan()

        if len(self.action_queue) > 0:

            current_action = self.action_queue[0]

            self.perform_action_save_memory(current_action)

            while len(self.action_queue) > 0 and self.action_queue[0].instant:
                current_action = self.action_queue[0]

                self.perform_action_save_memory(current_action)

        self.update_decision_model()

    def need_to_update_plan(self):
        return len(self.action_queue) == 0

    def plan(self):

        if self.plan_callable is not None:
            self.plan_callable(self)
            return

    def queue_action(self, action):

        if type(action) in self.memorize_tasks:
            self.private_learning_memory.save_state(self.get_features(type(action)), action)
            self.public_memory.save_state(self.get_features(type(action)), action)

        self.action_queue.append(action)

    def perform_action_save_memory(self, action):
        self.chosen_action = action

        if type(action) in self.memorize_tasks:
            results = self.perform_action(action)
            if results["done"]:
                self.private_learning_memory.save_results(self.get_target(type(action)), action)
                self.public_memory.save_results(self.get_target(type(action)), action)
        else:
            results = self.perform_action(action)

        self.chosen_action = None
        return results

    def perform_action(self, action):
        results = action.do_results()

        if results["done"] or not action.action_possible():
            self.action_log.append(self.action_queue.pop(0))

        return results

    def update_decision_model(self):
        model_to_use = None
        memory_to_use = None

        if self.memory_type == "public":
            memory_to_use = self.public_memory
        elif self.memory_type == "private":
            memory_to_use = self.private_learning_memory

        if self.model_type == "public":
            model_to_use = self.public_decision_model
        elif self.model_type == "private":
            model_to_use = self.private_decision_model

        if memory_to_use is None or model_to_use is None:
            raise Exception("You should set memory and model types ('public' or 'private')")
        #action_library.GoMating
        table_list = memory_to_use.make_table(action_library.GoMating)
        if len(table_list) >= self.memory_batch_size:
            df_train = np.asarray(table_list)
            # print df_train
            target_column = len(table_list[0])-1
            unique_targets = np.unique(df_train[:, target_column])  # TODO maybe discard
            if len(unique_targets) > 1:
                y_train = df_train[:, [target_column]].ravel()
                X_train = np.delete(df_train, target_column, 1)
                model_to_use.fit(X_train, y_train)
                memory_to_use.obliviate()
                print("Update successful")
            else:
                memory_to_use.obliviate()
                print("Memory discarded")

    def set_memorize_task(self, action_types, features_list, target):
        if isinstance(action_types, list):
            for action_type in action_types:
                self.memorize_tasks[action_type] = {"features": features_list,
                                                    "target": target}
        else:
            self.memorize_tasks[action_types] = {"features": features_list,
                                                 "target": target}

    def get_features(self, action_type):
        if action_type not in self.memorize_tasks:
            return None

        features_list_raw = self.memorize_tasks[action_type]["features"]
        features_list = []

        for feature_raw in features_list_raw:
            if isinstance(feature_raw, dict):
                if "kwargs" in feature_raw:
                    features_list.append(feature_raw["func"](**feature_raw["kwargs"]))
                else:
                    features_list.append(feature_raw["func"]())
            elif isinstance(feature_raw, collections.Callable):
                features_list.append(feature_raw())
            else:
                features_list.append(feature_raw)

        return features_list

    def get_target(self, action_type):
        if action_type not in self.memorize_tasks:
            return None

        target_raw = self.memorize_tasks[action_type]["target"]
        #collections.Callable
        if isinstance(target_raw,  collections.abc.Callable):
            return target_raw()
        elif isinstance(target_raw, dict):
            if "kwargs" in target_raw:
                return target_raw["func"](**target_raw["kwargs"])
            else:
                return target_raw["func"]()
        else:
            return target_raw


class Creature(Agent):
    def __init__(self):
        super(Creature, self).__init__()
        self.passable = False
        self.scenery = False
        self.alive = True
        self.name = ''
        self.sex = random.choice([True, False])
        if self.sex:
            self.color = "#550000"
        else:
            self.color = "#990000"
        self.mortal = True
        self.private_learning_memory = LearningMemory(self)
        self.public_memory = None

        self.private_decision_model = None
        self.public_decision_model = None

        self.plan_callable = None

        self.memory_type = ""
        self.model_type = ""

        self.memory_batch_size = 1

        self.memorize_tasks = {}
        self.chosen_action = None

    def __str__(self):
        return '@'

    @classmethod
    def class_name(cls):
        return "Creature"

    def pre_actions(self):
        if (self.time_of_death is not None) and self.z - self.time_of_death > 10:
            self.dissolve()
            return False

        if not self.alive:
            return False

        if random.random() <= 0.001 and self.age > 10:
            self.die()
            return False

        return True

    def die(self):
        if not self.mortal:
            return
        self.alive = False
        self.time_of_death = self.z

    def set_sex(self, sex):
        self.sex = sex
        if self.sex:
            self.color = "#550000"
        else:
            self.color = "#990000"

    def can_mate(self, with_who):
        if isinstance(with_who, Creature):
            if with_who.sex != self.sex:

                if not self.alive or not with_who.alive:
                    return False

                if self.sex:
                    return not with_who.has_state(Pregnant)
                else:
                    return not self.has_state(Pregnant)

        return False

    def will_mate(self, with_who):
        if not self.can_mate(with_who):
            return False

        if self.sex:
            if self.has_state(NotTheRightMood):
                return False
            return True
        else:
            self_has_substance = self.count_substance_of_type(Substance)
            partner_has_substance = with_who.count_substance_of_type(Substance)
            if self_has_substance + partner_has_substance == 0:
                return False
            if self_has_substance <= partner_has_substance:
                return True
            else:
                return random.random() < 1. * partner_has_substance / (self_has_substance*3 + partner_has_substance)

