"""Optional causal limits on motion, in portrait-relative coordinates.

These constrain model outputs; they do not repair learned decoder textures.
Defaults are experimental and disabled unless explicitly selected by the caller.
"""
import copy
import numpy as np


class MotionBounds:
    def __init__(self, source, pose_degrees=8., pose_step=2.,
                 lip_radius=.04, eye_radius=.025, expression_step=.015):
        values = (pose_degrees, pose_step, lip_radius, eye_radius, expression_step)
        if source is None or not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError("Motion bounds require a source and positive finite limits")
        self.source = copy.deepcopy(source)
        self.pose_degrees, self.pose_step = pose_degrees, pose_step
        self.lip_radius, self.eye_radius = lip_radius, eye_radius
        self.expression_step = expression_step
        self.previous = None

    @staticmethod
    def limit_vectors(value, radius):
        length = np.linalg.norm(value, axis=-1, keepdims=True)
        return value * np.minimum(1, radius / np.maximum(length, 1e-8))

    def __call__(self, driving):
        from .motion_stitch import bin66_to_degree
        result = copy.deepcopy(driving)
        for key in ('pitch', 'yaw', 'roll'):
            origin = bin66_to_degree(self.source[key])
            angle = bin66_to_degree(result[key])
            angle = np.clip(angle, origin - self.pose_degrees, origin + self.pose_degrees)
            if self.previous is not None:
                angle = np.clip(angle, self.previous[key] - self.pose_step,
                                self.previous[key] + self.pose_step)
            result[key] = angle
        origin = self.source['exp'].reshape(1, 21, 3)
        delta = result['exp'].reshape(1, 21, 3) - origin
        for indices, radius in (([6, 12, 14, 17, 19, 20], self.lip_radius),
                                ([11, 13, 15, 16, 18], self.eye_radius)):
            delta[:, indices] = self.limit_vectors(delta[:, indices], radius)
        if self.previous is not None:
            previous = self.previous['exp'].reshape(1, 21, 3) - origin
            delta = previous + self.limit_vectors(delta - previous, self.expression_step)
        result['exp'] = (origin + delta).reshape(1, -1)
        self.previous = copy.deepcopy(result)
        return result
