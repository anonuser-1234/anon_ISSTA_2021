


class Config:
    GEN_RANDOM = 'GEN_RANDOM'
    GEN_RANDOM_SEEDED = 'GEN_RANDOM_SEEDED'
    GEN_SEQUENTIAL_SEEDED = 'GEN_SEQUENTIAL_SEEDED'
    GEN_DIVERSITY = 'GEN_DIVERSITY'

    SEG_LENGTH = 25
    NUM_SPLINE_NODES =10
    INITIAL_NODE = (0.0, 0.0, -28.0, 8.0)
    ROAD_BBOX_SIZE = (-250, 0, 250, 500)
    EXECTIME = 0

    def __init__(self):
        self.experiment_name = 'exp'
        self.fitness_weights = (-1.0,)

        self.POPSIZE = 24
        self.POOLSIZE = 40
        self.NUM_GENERATIONS = 200000000000

        self.ARCHIVE_THRESHOLD = 35.0

        self.RESEED_UPPER_BOUND = int(self.POPSIZE * 0.1)

        self.MUTATION_EXTENT = 6.0
        self.MUTPB = 0.7
        self.simulation_save = True
        self.simulation_name = 'beamng_nvidia_runner/sim_$(id)'

        #self.keras_model_file = 'self-driving-car-4600.h5'
        self.keras_model_file = 'self-driving-car-185-2020.h5'

        # self.generator_name = Config.GEN_RANDOM
        # self.generator_name = Config.GEN_RANDOM_SEEDED
        #self.generator_name = Config.GEN_SEQUENTIAL_SEEDED
        self.generator_name = Config.GEN_DIVERSITY
        #self.seed_folder = 'population_HQ1'
        self.seed_folder = 'initial_pool'
        self.initial_population_folder = "initial_population"

        self.Feature_Combination = 8
        self.RUNTIME = 36000
        self.INTERVAL = 3600




