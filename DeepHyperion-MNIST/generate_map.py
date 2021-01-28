import operator
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
import json

# local imports
from individual import Individual
from plot_utils import plot_heatmap, plot_fives
import utils
from properties import RUNTIME, INTERVAL, NGEN, POPSIZE, EXPECTED_LABEL


def generate_maps(paths):
    for dir_path in paths:
        now = datetime.now().strftime("%Y%m%d%H%M%S")

        log_dir_name = f"dir_path/log_{now}"

        

        runs = sorted(glob.glob(f"{roads_path}/**/*.json", recursive=True),key=os.path.getmtime)
        for run in runs:
            individuals = []
            jsons = [f for f in glob.glob(f"{roads_path}/**/*.json", recursive=True) if "mbr" in f]
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    xml_desc = data["xml_desc"]
                    seed = int(data["seed"])
                    Digit(xml_desc, EXPECTED_LABEL)
                    individual: Individual = Individual(Digit, seed)
                    individuals.append(individual)


            for i in range(0,4):
                map_E = MapElitesMNIST(i, NGEN, POPSIZE, log_dir_name, True)
                # Create another folder insider the log one ...
                log_dir_path = Path(f"{log_dir_name}/{map_E.feature_dimensions[1].name}_{map_E.feature_dimensions[0].name}")
                log_dir_path.mkdir(parents=True, exist_ok=True)
                
                for ind in individuals:
                    map_E.place_in_mapelites(ind)

                # filled values                                 
                filled = np.count_nonzero(map_E.solutions != None)
                total = np.size(map_E.solutions)        

                original_seeds = set()
                mis_seeds = set()
                for (i, j), value in np.ndenumerate(map_E.solutions):
                    if map_E.solutions[i, j] is not None:
                        original_seeds.add(map_E.solutions[i, j].seed)
                        if map_E.performances[i, j] < 0:
                            mis_seeds.add(map_E.solutions[i, j].seed)

                Individual.COUNT_MISS = 0
                for (i, j), value in np.ndenumerate(map_E.performances):
                    if map_E.performances[i, j] < 0:
                        Individual.COUNT_MISS += 1
                        utils.print_image(f"{log_dir_path}/({i},{j})", map_E.solutions[i, j].member.purified, '')
                    elif 0 < map_E.performances[i, j] < np.inf:
                        utils.print_image(f"{log_dir_path}/({i},{j})", map_E.solutions[i, j].member.purified, 'gray')

                report = {
                    'Covered seeds': len(original_seeds),
                    'Filled cells': (filled),
                    'Filled density': (filled / total),
                    'Misclassified seeds': len(mis_seeds),
                    'Misclassification': (Individual.COUNT_MISS),
                    'Misclassification density': (Individual.COUNT_MISS / filled),
                }
                
                dst = f"{log_dir_name}/report_" + map_E.feature_dimensions[1].name + "_" + map_E.feature_dimensions[
                    0].name + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()
                
                repo = {
                    f"{map_E.feature_dimensions[1].name}_min": map_E.feature_dimensions[1].min,
                    f"{map_E.feature_dimensions[1].name}_max": map_E.feature_dimensions[1].bins,
                    f"{map_E.feature_dimensions[0].name}_min": map_E.feature_dimensions[0].min,
                    f"{map_E.feature_dimensions[0].name}_max": map_E.feature_dimensions[0].bins,
                    "Performances": map_E.performances.tolist()
                }
                filename = f"{log_dir_name}/results_{map_E.feature_dimensions[1].name}_{map_E.feature_dimensions[0].name}.json"
                with open(filename, 'w') as f:
                    f.write(json.dumps(repo))


def generate_rescaled_maps(log_dir_name, paths):
    min_bitmaps = np.inf
    min_orientation = np.inf
    min_moves = np.inf

    max_bitmaps = 0
    max_orientation = 0
    max_moves = 0

    jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "results_Bitmaps_Moves" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_Bitmaps > data["Bitmaps_min"]:
                min_Bitmaps = data["Bitmaps_min"]
            if max_Bitmaps < data["Bitmaps_max"]:
                max_Bitmaps = data["Bitmaps_max"]

            if min_Moves > data["Moves_min"]:
                min_Moves = data["Moves_min"]
            if max_Moves < data["Moves_max"]:
                max_Moves = data["Moves_max"]
            

    jsons = [g for g in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "results_Moves_Orientation" in g]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_Bitmaps > data["Moves_min"]:
                min_Bitmaps = data["Moves_min"]
            if max_Bitmaps < data["Moves_max"]:
                max_Bitmaps = data["Moves_max"]

            if min_Orientation > data["Orientation_min"]:
                min_Orientation = data["Orientation_min"]
            if max_Orientation < data["Orientation_max"]:
                max_Orientation = data["Orientation_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_Bitmaps_Orientation" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_Moves > data["Bitmaps_min"]:
                min_Moves = data["Bitmaps_min"]
            if max_Moves < data["Bitmaps_max"]:
                max_Moves = data["Bitmaps_max"]

            if min_Orientation > data["Orientation_min"]:
                min_Orientation = data["Orientation_min"]
            if max_Orientation < data["Orientation_max"]:
                max_Orientation = data["Orientation_max"]


    
    print(min_bitmaps)
    print(min_moves)
    print(min_orientation)


    print(max_bitmaps)
    print(max_moves)
    print(max_orientation)

    
    for path in paths:
        jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "results_Bitmaps_Moves" in f]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)

                fts = list()

                ft7 = FeatureDimension(name="Moves", feature_simulator="move_distance", bins=10)
                fts.append(ft7)

                # feature 2: Number of bitmaps above threshold
                ft2 = FeatureDimension(name="Bitmaps", feature_simulator="bitmap_count", bins=180)
                fts.append(ft2)


                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Bitmaps, max_Bitmaps, min_Moves, max_Moves)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misclassification': str(COUNT_MISS),
                    'Misclassification density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()



        jsons = [g for g in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "results_Moves_Orientation" in g]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)
                fts = list()

                # feature 1: orientation
                ft8 = FeatureDimension(name="Orientation", feature_simulator="orientation_calc", bins=100)
                fts.append(ft8)

                # feature 2: moves in svg path
                ft7 = FeatureDimension(name="Moves", feature_simulator="move_distance", bins=10)
                fts.append(ft7)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Moves, max_Moves, min_Orientation, max_Orientation)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misclassification': str(COUNT_MISS),
                    'Misclassification density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()


        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "results_Bitmaps_Orientation" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)
                fts = list()

                # feature 1: orientation
                ft8 = FeatureDimension(name="Orientation", feature_simulator="orientation_calc", bins=100)
                fts.append(ft8)

                # feature 2: Number of bitmaps above threshold
                ft2 = FeatureDimension(name="Bitmaps", feature_simulator="bitmap_count", bins=180)
                fts.append(ft2)
                

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Bitmaps, max_Bitmaps, min_Orientation, max_Orientation)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misclassification': str(COUNT_MISS),
                    'Misclassification density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()



        
        generate_reports(path.replace("/", "_"), path)

         

def generate_reports(filename, log_dir_name):
    filename = filename + ".csv"
    fw = open(filename, 'w')
    cf = csv.writer(fw, lineterminator='\n')
    # write the header
    cf.writerow(["Features", "Filled cells", "Filled density", 
                 "Misbehaviour", "Misbehaviour density"])

    jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_MeanLateralPosition_MinRadius" in f]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    cf.writerow(["MeanLateralPosition,MinRadius", str(np.mean(filled_cells)), str(np.mean(filled_density)), str(np.mean(misbehaviour)), str(np.mean(misbehaviour_density)) ])

    jsons = [g for g in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_DirectionCoverage_MinRadius" in g]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    cf.writerow(["DirectionCoverage,MinRadius", str(np.mean(filled_cells)), str(np.mean(filled_density)), str(np.mean(misbehaviour)), str(np.mean(misbehaviour_density)) ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_DirectionCoverage" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    cf.writerow(["MeanLateralPosition,DirectionCoverage", str(np.mean(filled_cells)), str(np.mean(filled_density)), str(np.mean(misbehaviour)), str(np.mean(misbehaviour_density)) ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_MinRadius" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    cf.writerow(["SegmentCount,MinRadius", str(np.mean(filled_cells)), str(np.mean(filled_density)), str(np.mean(misbehaviour)), str(np.mean(misbehaviour_density)) ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_SegmentCount" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    cf.writerow(["MeanLateralPosition,SegmentCount", str(np.mean(filled_cells)), str(np.mean(filled_density)), str(np.mean(misbehaviour)), str(np.mean(misbehaviour_density)) ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_DirectionCoverage" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    cf.writerow(["SegmentCount,DirectionCoverage", str(np.mean(filled_cells)), str(np.mean(filled_density)), str(np.mean(misbehaviour)), str(np.mean(misbehaviour_density)) ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_DirectionCoverage_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    cf.writerow(["DirectionCoverage,SDSteeringAngle", str(np.mean(filled_cells)), str(np.mean(filled_density)), str(np.mean(misbehaviour)), str(np.mean(misbehaviour_density)) ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MinRadius_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    cf.writerow(["MinRadius,SDSteeringAngle", str(np.mean(filled_cells)), str(np.mean(filled_density)), str(np.mean(misbehaviour)), str(np.mean(misbehaviour_density)) ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    cf.writerow(["SegmentCount,SDSteeringAngle", str(np.mean(filled_cells)), str(np.mean(filled_density)), str(np.mean(misbehaviour)), str(np.mean(misbehaviour_density)) ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    cf.writerow(["MeanLateralPosition,SDSteeringAngle", str(np.mean(filled_cells)), str(np.mean(filled_density)), str(np.mean(misbehaviour)), str(np.mean(misbehaviour_density)) ])


 
if __name__ == "__main__":
    generate_maps(["All-30/DeepJanus-30", "All-30/DLFuzz-30"])
    generate_rescaled_maps("All-30", ["All-30/DeepJanus-30", "All-30/DLFuzz-30"])  