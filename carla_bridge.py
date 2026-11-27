"""
CARLA bridge for CONTEXTRA (runs as its own process).

    python carla_bridge.py --spawn --weather HardRainNight
    python carla_bridge.py --attach          # use an existing vehicle with role_name "hero"

Reads speed, weather, road curvature and nearby traffic from a running CARLA
server and writes them to context_state.json, which the dashboard reads when
"Context source" is set to "CARLA bridge (file)".

STATUS: written against the CARLA 0.9.x Python API but NOT tested on a real CARLA
server (the development laptop cannot run CARLA). Test it first on a machine that
can, and do not claim it was demonstrated in CARLA until it has been.

Because it only talks to the dashboard through a JSON file, it can run in a
separate Python environment (the `carla` pip wheel supports only certain Python
versions - check which ones for your CARLA release) or even on another computer
sharing the project folder.

Mappings (see context_provider.py): visibility from fog + rain, curvature from
heading change ~20 m ahead, roughness from vertical acceleration (a crude proxy),
traffic from vehicles within 50 m, night flag from sun altitude.
"""

import argparse
import math
import random
import time

from context_provider import (DrivingContext, curvature_from_yaw_delta,
                              hour_from_sun_altitude, read_context_file,
                              roughness_from_vertical_accel,
                              traffic_from_nearby_vehicles,
                              visibility_from_weather, write_context_file)


def find_hero(world):
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == "hero":
            return actor
    return None


def spawn_vehicle(world):
    blueprint = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    blueprint.set_attribute("role_name", "hero")
    points = world.get_map().get_spawn_points()
    random.shuffle(points)
    for point in points:
        vehicle = world.try_spawn_actor(blueprint, point)
        if vehicle is not None:
            vehicle.set_autopilot(True)
            return vehicle
    raise RuntimeError("Could not spawn a vehicle (all spawn points blocked).")


def follow_with_spectator(world, vehicle):
    t = vehicle.get_transform()
    fwd = t.get_forward_vector()
    import carla
    loc = carla.Location(t.location.x - 8 * fwd.x, t.location.y - 8 * fwd.y, t.location.z + 4)
    world.get_spectator().set_transform(carla.Transform(loc, carla.Rotation(pitch=-18, yaw=t.rotation.yaw)))


def read_context(world, vehicle, start, minutes_offset):
    v = vehicle.get_velocity()
    speed_kmh = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    weather = world.get_weather()
    visibility = visibility_from_weather(weather.fog_density, weather.precipitation)
    hour = hour_from_sun_altitude(weather.sun_altitude_angle)

    carla_map = world.get_map()
    wp = carla_map.get_waypoint(vehicle.get_location())
    ahead = wp.next(20.0)
    curvature = 0.0
    if ahead:
        curvature = curvature_from_yaw_delta(
            ahead[0].transform.rotation.yaw - wp.transform.rotation.yaw)

    roughness = roughness_from_vertical_accel(vehicle.get_acceleration().z)

    here = vehicle.get_location()
    nearby = sum(1 for a in world.get_actors().filter("vehicle.*")
                 if a.id != vehicle.id and a.get_location().distance(here) <= 50.0)

    return DrivingContext(
        speed_kmh=speed_kmh, road_curvature=curvature, road_roughness=roughness,
        visibility=visibility, traffic_density=traffic_from_nearby_vehicles(nearby),
        hour_of_day=hour, drive_minutes=minutes_offset + (time.time() - start) / 60.0,
        source="carla")


def main():
    parser = argparse.ArgumentParser(description="CONTEXTRA CARLA bridge")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--spawn", action="store_true", help="spawn an autopilot vehicle")
    parser.add_argument("--attach", action="store_true", help="attach to an existing vehicle with role_name 'hero'")
    parser.add_argument("--weather", default=None, help="CARLA weather preset, e.g. ClearNoon, HardRainNight, SoftRainSunset")
    parser.add_argument("--minutes-offset", type=float, default=0.0, help="pretend the driver has already driven this long")
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    import carla  # imported here so the rest of CONTEXTRA does not need the carla package

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    if args.weather:
        preset = getattr(carla.WeatherParameters, args.weather, None)
        if preset is None:
            raise SystemExit(f"Unknown weather preset: {args.weather}")
        world.set_weather(preset)

    spawned = None
    vehicle = find_hero(world) if (args.attach or not args.spawn) else None
    if vehicle is None:
        if not args.spawn:
            raise SystemExit("No 'hero' vehicle found. Use --spawn to create one.")
        vehicle = spawned = spawn_vehicle(world)

    print("CARLA bridge running. Ctrl+C to stop.")
    start = time.time()
    try:
        while True:
            ctx = read_context(world, vehicle, start, args.minutes_offset)
            write_context_file(ctx)
            if spawned is not None:
                follow_with_spectator(world, vehicle)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if spawned is not None:
            spawned.destroy()


if __name__ == "__main__":
    main()
