#!/usr/bin/env python
# -*- coding:utf-8 -*-

from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
from serpent.input_controller import MouseButton

import serpent.cv
import serpent.utilities

from serpent.frame_grabber import FrameGrabber

from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace

import time
import sys
import collections
import os
import gc
import _thread

import numpy as np

import skimage.io
import skimage.filters
import skimage.morphology
import skimage.measure
import skimage.draw
import skimage.segmentation
import skimage.color

import pyperclip
from datetime import datetime
from .helpers.frame_processing import frame_to_hearts
import cv2

try:
    import Image
except ImportError:
    from PIL import Image

import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'D:\\Program Files (x86)\\Tesseract-OCR\\tesseract'


class SerpenthornetGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.analytics_client = None
        self.game_state = None

        self.s_p1 = 16
        self.game_over = False
        self.reward = 0
        self._reset_game_state()

    def setup_play(self):
        input_mapping = {
            "UP": [KeyboardKey.KEY_W],
            "LEFT": [KeyboardKey.KEY_A],
            "DOWN": [KeyboardKey.KEY_S],
            "RIGHT": [KeyboardKey.KEY_D],
            "LEFT_UP": [KeyboardKey.KEY_W, KeyboardKey.KEY_A],
            "RIGHT_UP": [KeyboardKey.KEY_W, KeyboardKey.KEY_D],
            "LEFT_DOWN": [KeyboardKey.KEY_S, KeyboardKey.KEY_A],
            "RIGHT_DOWN": [KeyboardKey.KEY_S, KeyboardKey.KEY_D],
            "UP_SHOOT": [KeyboardKey.KEY_UP],
            "LEFT_SHOOT": [KeyboardKey.KEY_LEFT],
            "DOWN_SHOOT": [KeyboardKey.KEY_DOWN],
            "RIGHT_SHOOT": [KeyboardKey.KEY_RIGHT],
            "LEFT_UP_SHOOT": [KeyboardKey.KEY_LEFT, KeyboardKey.KEY_UP],
            "RIGHT_UP_SHOOT": [KeyboardKey.KEY_RIGHT, KeyboardKey.KEY_UP],
            "LEFT_DOWN_SHOOT": [KeyboardKey.KEY_LEFT, KeyboardKey.KEY_DOWN],
            "RIGHT_DOWN_SHOOT": [KeyboardKey.KEY_RIGHT, KeyboardKey.KEY_DOWN],
            "BOOM": [KeyboardKey.KEY_SPACE]
        }

        self.key_mapping = {
            KeyboardKey.KEY_W.name: "MOVE UP",
            KeyboardKey.KEY_A.name: "MOVE LEFT",
            KeyboardKey.KEY_S.name: "MOVE DOWN",
            KeyboardKey.KEY_D.name: "MOVE RIGHT",
            KeyboardKey.KEY_UP.name: "SHOOT UP",
            KeyboardKey.KEY_LEFT.name: "SHOOT LEFT",
            KeyboardKey.KEY_DOWN.name: "SHOOT DOWN",
            KeyboardKey.KEY_RIGHT.name: "SHOOT RIGHT",
        }

        movement_action_space = KeyboardMouseActionSpace(
            directional_keys=[None, "UP", "LEFT", "DOWN", "RIGHT", "LEFT_UP", "RIGHT_UP", "LEFT_DOWN", "RIGHT_DOWN"]
        )

        projectile_action_space = KeyboardMouseActionSpace(
            projectile_keys=[None, "UP_SHOOT", "LEFT_SHOOT", "DOWN_SHOOT", "RIGHT_SHOOT", "LEFT_UP_SHOOT", "RIGHT_UP_SHOOT",
                             "LEFT_DOWN_SHOOT", "RIGHT_DOWN_SHOOT", "BOOM"]
        )

        movement_model_file_path = "datasets/hornet_movement_dqn.h5".replace("/", os.sep)

        self.dqn_movement = DDQN(
            model_file_path=movement_model_file_path if os.path.isfile(movement_model_file_path) else None,
            input_shape=(100, 100, 3),
            input_mapping=input_mapping,
            action_space=movement_action_space,
            replay_memory_size=5000,
            max_steps=1000000,
            observe_steps=10,
            batch_size=16,
            initial_epsilon=1,
            final_epsilon=0.01,
            override_epsilon=False
        )

        projectile_model_file_path = "datasets/hornet_projectile_dqn.h5".replace("/", os.sep)

        self.dqn_projectile = DDQN(
            model_file_path=projectile_model_file_path if os.path.isfile(projectile_model_file_path) else None,
            input_shape=(100, 100, 3),
            input_mapping=input_mapping,
            action_space=projectile_action_space,
            replay_memory_size=5000,
            max_steps=1000000,
            observe_steps=10,
            batch_size=16,
            initial_epsilon=1,
            final_epsilon=0.01,
            override_epsilon=False
        )
        # try:
        #     self.dqn_projectile.load_model_weights(file_path='model/hornet/binding_of_isaac_projectile_dqn_150000_0.5687282200080599_.h5',override_epsilon=True)
        #     self.dqn_movement.load_model_weights(file_path='model/hornet/binding_of_isacc_movement_dqn_150000_0.5687282200080599_.h5', override_epsilon=True)
        # except Exception as e:
        #     raise e

    def get_reward_state(self, heart, score):
        try:
            score = int(score)
            if score > self.game_state['game_score']:
                score_reward = 0.5
            elif self.game_state['game_score'] - score > 100:
                score_reward = 0.5
            else:
                score_reward = 0
            self.game_state['game_score'] = score
        except Exception as e:
            score_reward = 0

        if heart == -1:
            # print(heart, self.game_over, "restart game waiting")
            self.game_over = True
            self.reward = 0
            self.s_p1 = 16
        elif self.game_over == False and heart == 0:
            self.game_over = True
            self.reward = ((16 - (self.s_p1 - heart)*2) / 16) + score_reward
            self.s_p1 = 16
            # print(heart, self.game_over, self.reward)
            self.reward = 0
        elif self.game_over == True and heart == 0:
            pass
            # print(heart, self.game_over, "game over, restart waiting")
        elif heart != 0 and heart != -1:
            self.game_over = False
            self.reward = ((16 - (self.s_p1 - heart)*16) / 16) + score_reward
            self.s_p1 = heart
            # print(heart, self.game_over, self.reward)

    def handle_play(self, game_frame):
        gc.disable()
        heart1 = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions['HUD_HEART_15'])
        heart_p = heart1[0, 0, 0]
        print(heart_p)
        if heart_p == 0:
            gc.enable()
            gc.collect()
            gc.disable()
            for i, game_frame in enumerate(self.game_frame_buffer.frames):
                self.visual_debugger.store_image_data(
                    game_frame.frame,
                    game_frame.frame.shape,
                    str(i)
                )
            self.input_controller.tap_key(KeyboardKey.KEY_D)
            self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        else:
            self.train_ddqn(game_frame)

    def train_ddqn(self, game_frame):
        if self.dqn_movement.first_run:
            self.dqn_movement.first_run = False
            self.dqn_projectile.first_run = False
            return None

        heart = frame_to_hearts(game_frame.frame, self.game)
        score = self._process_ocr(game_frame)
        self.get_reward_state(heart, score)

        if self.dqn_movement.frame_stack is None:
            pipline_game_frame = FrameGrabber.get_frames(
                [0],
                frame_shape=(self.game.frame_height, self.game.frame_width),
                frame_type="PIPELINE",
                dtype="float64"
            ).frames[0]
            print(np.shape(pipline_game_frame.frame))
            # self.dqn_movement.build_frame_stack(pipline_game_frame.frame)

            self.dqn_movement.frame_stack = self._build_frame_stack(pipline_game_frame.frame)
            self.dqn_projectile.frame_stack = self.dqn_movement.frame_stack

        else:
            game_frame_buffer = FrameGrabber.get_frames(
                # [0, 4, 8, 12],
                [0],
                frame_shape=(self.game.frame_height, self.game.frame_width),
                frame_type="PIPELINE",
                dtype="float64"
            )

            if self.dqn_movement.mode == "TRAIN":
                self.game_state["run_reward_movement"] += self.reward
                self.game_state["run_reward_projectile"] += self.reward

                self._movement_append_to_replay_memory(
                    game_frame_buffer,
                    self.reward,
                    terminal=self.game_over
                )

                self._projectile_append_to_replay_memory(
                    game_frame_buffer,
                    self.reward,
                    terminal=self.game_over
                )

                #Every 2000 steps, save latest weights to disk
                if self.dqn_movement.current_step % 2000 == 0:
                    self.dqn_movement.save_model_weights(
                        file_path_prefix=f"datasets/binding_of_isacc_movement"
                    )
                    self.dqn_projectile.save_model_weights(
                        file_path_prefix=f"datasets/binding_of_isaac_projectile"
                    )

                #Every 20000 steps, save weights checkpoint to disk
                if self.dqn_movement.current_step % 20000 == 0:
                    self.dqn_movement.save_model_weights(
                        file_path_prefix=f"datasets/c_binding_of_isaac_movement",
                        is_checkpoint=True
                    )
                    self.dqn_projectile.save_model_weights(
                        file_path_prefix=f"datasets/c_binding_of_isaac_projectile",
                        is_checkpoint=True
                    )

            elif self.dqn_movement.mode == "RUN":
                game_frames = [game_frame.frame for game_frame in game_frame_buffer.frames]
                self.dqn_movement.frame_stack = np.array(game_frames)
                self.dqn_projectile.frame_stack = np.array(game_frames)

            run_time = datetime.now() - self.started_at
            serpent.utilities.clear_terminal()

            print(f"SESSION RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours,"
                  f" {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} seconds")

            print("MOVEMENT NEURAL NETWORK:\n")
            self.dqn_movement.output_step_data()
            print(f"reward:{self.reward}")

            print("PROJECTILE NEURAL NETWORK:\n")
            self.dqn_projectile.output_step_data()

            print(f"CURRENT RUN: {self.game_state['current_run']}")
            print(f"CURRENT RUN REWARD: "
                  f"{round(self.reward + self.reward, 2)}")

            print(f"CURRENT RUN PREDICTED ACTIONS: {self.game_state['run_predicted_actions']}")
            print(f"CURRENT HEALTH: {heart}")
            print(f"LAST RUN DURATION: {self.game_state['last_run_duration']} seconds")

            print(f"RECORD TIME ALIVE: {self.game_state['record_time_alive'].get('value')} seconds "
                  f"(Run {self.game_state['record_time_alive'].get('run')}, "
                  f"{'Predicted' if self.game_state['record_time_alive'].get('predicted') else 'Training'}")

            print(f"RANDOM AVERAGE TIME ALIVE: {self.game_state['random_time_alive']} seconds")


            if self.game_over == True:
                serpent.utilities.clear_terminal()
                timestamp = datetime.utcnow()

                gc.enable()
                gc.collect()
                gc.disable()

                timestamp_delta = timestamp - self.game_state["run_timestamp"]
                self.game_state["last_run_duration"] = timestamp_delta.seconds

                if self.dqn_movement.mode in ["TRAIN", "RUN"]:
                    #Check for Records
                    if self.game_state["last_run_duration"] > self.game_state["record_time_alive"].get("value", 0):
                        self.game_state["record_time_alive"] = {
                            "value": self.game_state["last_run_duration"],
                            "run": self.game_state["current_run"],
                            "predicted": self.dqn_movement.mode == "RUN"
                        }

                else:
                    self.game_state["random_time_alives"].append(self.game_state["last_run_duration"])
                    self.game_state["random_time_alive"] = np.mean(self.game_state["random_time_alives"])

                self.game_state["current_run_state"] = 0
                self.input_controller.handle_keys([])

                if self.dqn_movement.mode == "TRAIN":
                    for i in range(16):
                        serpent.utilities.clear_terminal()
                        print(f"TRAINING ON MINI-BATCHES: {i + 1}/16")
                        print(f"NEXT RUN: {self.game_state['current_run'] + 1} "
                              f"{'- AI RUN' if (self.game_state['current_run'] + 1) % 20 == 0 else ''}")

                        self.dqn_movement.train_on_mini_batch()
                        self.dqn_projectile.train_on_mini_batch()

                self.game_state["run_timestamp"] = datetime.utcnow()
                self.game_state["current_run"] += 1
                self.game_state["run_reward_movement"] = 0
                self.game_state["run_reward_projectile"] = 0
                self.game_state["run_predicted_actions"] = 0
                self.s_p1 = 16
                self.game_over = False
                self.reward = 0

                if self.dqn_movement.mode in ["TRAIN", "RUN"]:
                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 100 == 0:
                        self.dqn_movement.update_target_model()
                        self.dqn_projectile.update_target_model()

                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 20 == 0:
                        self.dqn_movement.enter_run_mode()
                        self.dqn_projectile.enter_run_mode()
                    else:
                        self.dqn_movement.enter_train_mode()
                        self.dqn_projectile.enter_train_mode()

                return None

        self.dqn_movement.pick_action()
        self.dqn_movement.generate_action()

        self.dqn_projectile.pick_action(action_type=self.dqn_movement.current_action_type)
        self.dqn_projectile.generate_action()

        try:
            _thread.start_new_thread(self._execute_action, ("Thread", ))
        except Exception as e:
            print(e)

        if self.dqn_movement.current_action_type == "PREDICTED":
            self.game_state["run_predicted_actions"] += 1

        self.dqn_movement.erode_epsilon(factor=2)
        self.dqn_projectile.erode_epsilon(factor=2)

        self.dqn_movement.next_step()
        self.dqn_projectile.next_step()

        self.game_state["current_run_steps"] += 1

    def _execute_action(self, threadname):
        movement_keys = self.dqn_movement.get_input_values()
        projectile_keys = self.dqn_projectile.get_input_values()
        self.input_controller.handle_keys(movement_keys + projectile_keys)

    def _reset_game_state(self):
        self.game_state = {
            "current_run": 19,
            "current_run_steps": 0,
            "run_reward_movement": 0,
            "run_reward_projectile": 0,
            "run_future_rewards": 0,
            "run_predicted_actions": 0,
            "run_timestamp": datetime.utcnow(),
            "last_run_duration": 0,
            "record_time_alive": dict(),
            "random_time_alive": None,
            "random_time_alives": list(),
            "game_score": 0
        }

    def _process_ocr(self, game_frame):
        score_image = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["SCORE"])
        score_image = cv2.cvtColor(score_image, cv2.COLOR_BGR2GRAY)
        score_image[score_image < 255] = 0
        score_image = Image.fromarray(np.uint8(score_image))
        # score_image.show()
        # print(np.shape(score))
        # print(pytesseract.image_to_string(score_image, lang='chi_sim+eng'))
        return pytesseract.image_to_string(score_image,
                                           lang='chi_sim',
                                           boxes=False,
                                           config='--psm 10 --eom 3 -c tessedit_char_whitelist=0123456789')


    def _build_frame_stack(self, game_frame):
        frame_stack = np.stack((
            game_frame
        ), axis=0)
        return frame_stack.reshape((1,) + frame_stack.shape)

    def _movement_append_to_replay_memory(self, game_frame_buffer, reward, terminal=False):
        game_frames = [game_frame.frame for game_frame in game_frame_buffer.frames]
        previous_frame_stack = self.dqn_movement.frame_stack
        self.dqn_movement.frame_stack = np.array(game_frames)

        observation = [
            previous_frame_stack,
            self.dqn_movement.current_action_index,
            reward,
            self.dqn_movement.frame_stack,
            terminal
        ]
        self.dqn_movement.replay_memory.add(self.dqn_movement.calculate_target_error(observation), observation)

    def _projectile_append_to_replay_memory(self, game_frame_buffer, reward, terminal=False):
        game_frames = [game_frame.frame for game_frame in game_frame_buffer.frames]
        previous_frame_stack = self.dqn_projectile.frame_stack
        self.dqn_projectile.frame_stack = np.array(game_frames)

        observation = [
            previous_frame_stack,
            self.dqn_projectile.current_action_index,
            reward,
            self.dqn_projectile.frame_stack,
            terminal
        ]
        self.dqn_projectile.replay_memory.add(self.dqn_projectile.calculate_target_error(observation), observation)