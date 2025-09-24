from abc import ABC, abstractmethod
import os
import numpy as np
import importlib
import yaml
import torch
from typing import Optional
from scipy.linalg import block_diag


class Robot(ABC):
    def __init__(self, prefix):
        self.prefix: str = prefix
        self._mjcf_path = None
        self._urdf_path = None
        self._cfg_path = None

    @staticmethod
    def reverse_mapping(mapping):
        reverse = {}
        for key, values in mapping.items():
            if values is None:
                continue
            for value in values:
                if value in reverse:
                    reverse[value].append(key)
                else:
                    reverse[value] = [key]

        return reverse

    def get_file_path(self, type):
        if type == "mjcf":
            if self._mjcf_path is None:
                raise NameError("MJCF file does not exist.")
            return self._mjcf_path
        elif type == "urdf":
            if self._urdf_path is None:
                raise NameError("URDF file does not exist.")
            return self._urdf_path
        else:
            raise ValueError(f"Unsupported type {type}.")

    @property
    def cfg_path(self):
        if self._cfg_path is None:
            raise NameError("cfg file does not exist.")
        return self._cfg_path

    @property
    def base_pose(self):
        """
        [x, y, z, rx, ry, rz, rw]
        """
        return self._base_pose


# ---------------------------------------------------------------------------
class Hand(Robot, ABC):
    def __init__(self, prefix):
        """
        Args:
            prefix: the link/joint name will be '<prefix>_<name>'
        """
        super().__init__(prefix)

    @property
    def fingertip_names(self):
        return [f"{self.prefix}_{name}" for name in self._fingertip_names]

    @property
    def mano2dex_mapping(self):
        mano2dex_mapping = {}
        for key, val in self._mano2dex_mapping.items():
            mano2dex_mapping[key] = [f"{self.prefix}{name}" for name in val]
        return mano2dex_mapping

    @property
    def dex2mano_mapping(self):
        res = self.reverse_mapping(self.mano2dex_mapping)
        assert len(res.keys()) == len(self.body_names)
        return res

    @property
    def base_name(self):
        return f"{self.prefix}{self._base_name}"

    def to_dex(self, mano_body):
        return self.mano2dex_mapping[mano_body]

    def to_mano(self, dex_body):
        return self.dex2mano_mapping[dex_body]

    @property
    def dof_names(self):
        return [f"{self.prefix}{name}" for name in self._dof_names]

    @property
    def doa_names(self):
        return [f"{self.prefix}{name}" for name in self._doa_names]

    @property
    def body_names(self):
        return [f"{self.prefix}{name}" for name in self._body_names]

    @property
    def doa_max_vel(self):
        return self._doa_max_vel

    @property
    def n_dof(self):
        return len(self.dof_names)

    @property
    def n_doa(self):
        return len(self.doa_names)

    @property
    def n_bodies(self):
        return len(self.body_names)

    @property
    def doa2dof_matrix(self):
        return self._doa2dof_matrix.copy()

    @property
    def doa_kp(self):
        return self._doa_kp.copy()


# ---------------------------------------------------------------------------
class Arm(Robot, ABC):
    def __init__(self, prefix):
        """
        Args:
            prefix: the link/joint name will be '<prefix>_<name>'
        """
        super().__init__(prefix)

    @property
    def dof_names(self):
        return [f"{self.prefix}{name}" for name in self._dof_names]

    @property
    def doa_names(self):
        return [f"{self.prefix}{name}" for name in self._doa_names]

    @property
    def body_names(self):
        return [f"{self.prefix}{name}" for name in self._body_names]

    @property
    def doa_max_vel(self):
        return self._doa_max_vel

    @property
    def n_dof(self):
        return len(self.dof_names)

    @property
    def n_doa(self):
        return len(self.doa_names)

    @property
    def n_bodies(self):
        return len(self.body_names)

    @property
    def doa2dof_matrix(self):
        return self._doa2dof_matrix.copy()

    @property
    def doa_kp(self):
        return self._doa_kp.copy()


# ---------------------------------------------------------------------------
class ArmHand(Robot, ABC):
    def __init__(self, prefix):
        super().__init__(prefix)

        self.arm: Arm = None
        self.hand: Hand = None

    @property
    def dof_names(self):
        return (self.arm.dof_names if self.arm else []) + self.hand.dof_names

    @property
    def doa_names(self):
        return (self.arm.doa_names if self.arm else []) + self.hand.doa_names

    @property
    def body_names(self):
        return (self.arm.body_names if self.arm else []) + self.hand.body_names

    @property
    def doa_max_vel(self):
        return (self.arm.doa_max_vel if self.arm else []) + self.hand.doa_max_vel

    @property
    def n_dof(self):
        return len(self.dof_names)

    @property
    def n_doa(self):
        return len(self.doa_names)

    @property
    def n_bodies(self):
        return len(self.body_names)

    @property
    def base_name(self):
        return f"{self.prefix}{self.arm._base_name}"

    @property
    def doa2dof_matrix(self):
        if self.arm is None:
            return self.hand.doa2dof_matrix
        else:
            return block_diag(self.arm.doa2dof_matrix, self.hand.doa2dof_matrix)

    @property
    def doa_kp(self):
        return (self.arm.doa_kp if self.arm else []) + self.hand.doa_kp


# class DualArmHand(Robot, ABC):
#     def __init__(self):
#         self.left: ArmHand = None
#         self.right: ArmHand = None


# -------------------------------------------------------------------
# Robot Registration Helper
# -------------------------------------------------------------------
class RobotFactory:
    _registry = {}

    @classmethod
    def register(cls, robot_type: str, robot_class):
        """Register a new hand type."""
        cls._registry[robot_type] = robot_class

    @classmethod
    def create_robot(cls, robot_type: str, *args, **kwargs) -> Hand:
        if robot_type not in cls._registry:
            raise ValueError(f"Hand type '{robot_type}' not registered.")
        return cls._registry[robot_type](*args, **kwargs)

    @classmethod
    def auto_register_robots(cls, directory: str, base_package: str):
        """Automatically import all hand modules in the directory."""
        for filename in os.listdir(directory):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = f"{base_package}.{filename[:-3]}"
                importlib.import_module(module_name)


def register_robot(robot_type):
    def decorator(cls):
        RobotFactory.register(robot_type, cls)
        return cls

    return decorator
