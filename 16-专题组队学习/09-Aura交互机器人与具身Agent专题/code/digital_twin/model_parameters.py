"""One dimensional source; CAD uses mm, simulation converts to SI explicitly."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Parameters:
    body_rx: float = 120.
    body_ry: float = 85.
    body_rz: float = 94.
    body_z: float = 105.
    face_cut: float = 47.
    face_radius: float = 82.
    face_tilt_deg: float = 25.
    screen_radius: float = 58.
    screen_v: float = -13.
    camera_spacing: float = 78.
    camera_v: float = 49.
    wheel_radius: float = 72.
    wheel_width: float = 20.
    wheel_y: float = 95.
    wheel_x: float = -5.
    caster_x: float = -78.
    caster_radius: float = 12.
    chassis_mass_kg: float = 1.2
    wheel_mass_kg: float = .12

    @property
    def tilt(self):
        return math.radians(self.face_tilt_deg)

    @property
    def face_rotation(self):
        s, c = math.sin(self.tilt), math.cos(self.tilt)
        return [[0., -s, c], [1., 0., 0.], [0., c, s]]

    def face_point(self, u=0., v=0., depth=0.):
        s, c = math.sin(self.tilt), math.cos(self.tilt)
        return [(self.face_cut + depth)*c - v*s, u,
                self.body_z + (self.face_cut + depth)*s + v*c]

    def record(self):
        return {'values_mm_unless_suffixed': asdict(self),
                'fidelity': 'photo_based_provisional_not_metrology_1_to_1',
                'confirmed': {'official_overall_approx_cm': [21, 20, 20]},
                'unmeasured': ['all_part_dimensions', 'mass', 'inertia', 'friction',
                               'camera_intrinsics', 'display_active_diameter', 'face_tilt'],
                'official_source': 'https://frontierx.online/'}


P = Parameters()
