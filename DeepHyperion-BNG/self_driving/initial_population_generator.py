import sys
import os
from pathlib import Path

path = Path(os.path.abspath(__file__))
# This corresponds to DeepHyperion-BNG
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))

from core.folder_storage import SeedStorage
import glob, json
from random import shuffle, choice
from shutil import copy
from self_driving.road_bbox import RoadBoundingBox
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_individual import BeamNGIndividual
import core.utils as us
from scipy.spatial import distance
from core.feature_dimension import FeatureDimension


def get_spine(member):
    print("member: ", member)
    with open(member) as json_file:
        spine = json.load(json_file)
        return spine['sample_nodes']

def get_min_distance_from_set(ind, solution):
    distances = list()
    # print("ind:", ind)
    # print("solution:", solution)
    ind_spine = ind[0]

    for road in solution:
        road_spine = road[0]
        distances.append(manhattan_dist(ind_spine, road_spine))
    distances.sort()
    return distances[0]

def manhattan_dist(ind1, ind2):
    return distance.cityblock(list(ind1), list(ind2))


def initial_pool_generator(config, problem):
    good_members_found = 0
    attempts = 0
    storage = SeedStorage('initial_pool')

    while good_members_found < config.POOLSIZE:#40:
        path = storage.get_path_by_index(good_members_found + 1)
        # if path.exists():
        #     print('member already exists', path)
        #     good_members_found += 1
        #     continue
        attempts += 1
        print(f'attempts {attempts} good {good_members_found} looking for {path}')
        member = problem.generate_random_member()
        member.evaluate()
        if member.distance_to_boundary <= 0:
            continue
        member = problem.member_class().from_dict(member.to_dict())
        member.config = config
        member.problem = problem
        member.clear_evaluation()

        member.distance_to_boundary = None
        good_members_found += 1
        path.write_text(json.dumps(member.to_dict()))

    return storage.folder

# def initial_population_generator(path, config, problem):
#     all_roads = [filename for filename in glob.glob(str(path)+"\*.json", recursive=True)]
#     #all_roads += [filename for filename in glob.glob(path2)]
#
#     shuffle(all_roads)
#
#     roads = all_roads[:40]
#
#     starting_point = choice(roads)
#
#     original_set = list()
#     original_set.append(starting_point)
#
#     popsize = config.POPSIZE
#
#     i = 0
#     while i < popsize-1:
#         max_dist = 0
#         for ind in roads:
#
#             dist = get_min_distance_from_set(ind, original_set)
#             if dist > max_dist:
#                 max_dist = dist
#                 best_ind = ind
#         original_set.append(best_ind)
#         i += 1
#
#     base = config.initial_population_folder
#     storage = SeedStorage(base)
#     for index, road in enumerate(original_set):
#         path = storage.get_path_by_index(index + 1)
#         dst = path
#         copy(road,dst)

def initial_population_generator(path, config, problem):
    all_roads = [filename for filename in glob.glob(str(path)+"\*.json", recursive=True)]
    type = config.Feature_Combination
    shuffle(all_roads)

    roads = all_roads[:40]

    original_set = list()

    individuals = []
    popsize = config.POPSIZE
    i = 0
    for road in roads:
        with open(road) as json_file:
            data = json.load(json_file)
        sample_nodes = data["sample_nodes"]
        bbox_size = (-250.0, 0.0, 250.0, 500.0)
        road_bbox = RoadBoundingBox(bbox_size)
        res = BeamNGMember([], [tuple(t) for t in sample_nodes], len(sample_nodes), road_bbox)
        res.config = config
        res.problem = problem
        individual: BeamNGIndividual = BeamNGIndividual(res, config)
        individual.m.sample_nodes = us.new_resampling(sample_nodes)
        individual.evaluate()
        b = tuple()
        feature_dimensions = generate_feature_dimension(type)
        for ft in feature_dimensions:
            i = feature_simulator(ft.feature_simulator, individual)
            b = b + (i,)
        individuals.append([b, road, individual])

    starting_point = choice(individuals)
    original_set.append(starting_point)

    i = 0
    while i < popsize - 1:
        max_dist = 0
        for ind in individuals:
            dist = get_min_distance_from_set(ind, original_set)
            if dist > max_dist:
                max_dist = dist
                best_ind = ind
        original_set.append(best_ind)
        i += 1

    base = config.initial_population_folder
    storage = SeedStorage(base)
    for index, road in enumerate(original_set):
        dst = storage.get_path_by_index(index + 1)
        ind = road[2]
        #copy(road[1], dst)
        with open(road[1]) as ff:
            json_file = json.load(ff)
        with open(dst, 'w') as f:
            f.write(json.dumps({
                "control_nodes": json_file["control_nodes"],
                ind.m.simulation.f_params: ind.m.simulation.params._asdict(),
                ind.m.simulation.f_info: ind.m.simulation.info.__dict__,
                ind.m.simulation.f_road: ind.m.simulation.road.to_dict(),
                ind.m.simulation.f_records: [r._asdict() for r in ind.m.simulation.states]
            }))

def feature_simulator(function, x):
    """
    Calculates the number of control points of x's svg path/number of bitmaps above threshold
    :param x: genotype of candidate solution x
    :return:
    """
    if function == 'min_radius':
        return us.new_min_radius(x)
    if function == 'mean_lateral_position':
        return us.mean_lateral_position(x)
    if function == "dir_coverage":
        return us.direction_coverage(x)
    if function == "segment_count":
        return us.segment_count(x)
    if function == "sd_steering":
        return us.sd_steering(x)

def generate_feature_dimension(_type):
    fts = list()

    if _type == 0:
        ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
        fts.append(ft1)
        ft2 = FeatureDimension(name="DirectionCoverage", feature_simulator="dir_coverage", bins=1)
        fts.append(ft2)

    elif _type == 1:
        ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
        fts.append(ft1)

        ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
        fts.append(ft3)

    elif _type == 2:
        ft2 = FeatureDimension(name="DirectionCoverage", feature_simulator="dir_coverage", bins=1)
        fts.append(ft2)

        ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
        fts.append(ft3)

    elif _type == 3:
        ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
        fts.append(ft1)
        ft2 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
        fts.append(ft2)

    elif _type == 4:
        ft1 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
        fts.append(ft1)

        ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
        fts.append(ft3)

    elif _type == 5:
        ft2 = FeatureDimension(name="DirectionCoverage", feature_simulator="dir_coverage", bins=1)
        fts.append(ft2)

        ft3 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
        fts.append(ft3)

    elif _type == 6:
        ft2 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=1)
        fts.append(ft2)

        ft3 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
        fts.append(ft3)

    elif _type == 7:
        ft2 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=1)
        fts.append(ft2)

        ft3 = FeatureDimension(name="DirectionCoverage", feature_simulator="dir_coverage", bins=1)
        fts.append(ft3)

    elif _type == 8:
        ft2 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=1)
        fts.append(ft2)

        ft3 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
        fts.append(ft3)

    elif _type == 9:
        ft2 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=1)
        fts.append(ft2)

        ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
        fts.append(ft3)

    return fts


if __name__ == '__main__':
    path = initial_pool_generator()
    #path = r"C:\Users\Aurora\new-DeepJanus\DeepJanus\DeepJanus-BNG\data\member_seeds\initial_pool"
    initial_population_generator(path)