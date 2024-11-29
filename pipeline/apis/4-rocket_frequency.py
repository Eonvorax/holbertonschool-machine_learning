#!/usr/bin/env python3
"""
Display the number of SpaceX launches per rocket
"""
import requests
from collections import defaultdict


def main():
    """
    Displays the number of SpaceX launches per rocket
    """
    # Fetch all launches
    launches_url = "https://api.spacexdata.com/v4/launches"
    launches_response = requests.get(launches_url).json()

    # Fetch all rockets
    rockets_url = "https://api.spacexdata.com/v4/rockets"
    rockets_response = requests.get(rockets_url).json()

    # Map the rocket IDs to names
    rocket_id_to_name = {rocket['id']: rocket['name']
                         for rocket in rockets_response}

    # Count launches per rocket
    rocket_launch_counts = defaultdict(int)
    for launch in launches_response:
        rocket_id = launch['rocket']
        if rocket_id in rocket_id_to_name:
            rocket_launch_counts[rocket_id_to_name[rocket_id]] += 1

    # Sort rockets by launch count (descending), then alphabetically by name
    sorted_rockets = sorted(
        rocket_launch_counts.items(),
        key=lambda x: (-x[1], x[0])
    )
    # NOTE Negative x[1] ensures the items are sorted in descending order of
    # launch count (highest to lowest).

    for rocket_name, launch_count in sorted_rockets:
        print(f"{rocket_name}: {launch_count}")


if __name__ == "__main__":
    main()
