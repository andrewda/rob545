import math
import sys
import time

import neat
import pygame
import pickle

import numpy as np


import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(suppress=True)

import matplotlib.pyplot as plt

import os
import copy


# Constants
# WIDTH = 1600
# HEIGHT = 880

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 40
CAR_SIZE_Y = 40

DEATH_ZONE = (255, 255, 255, 255)  # Color To Crash on Hit
SAFE_ZONE = (200, 200, 0, 255)  # Color of Safe Zone
UNSAFE_ZONE = (255, 255, 16, 255)  # Color of Unsafe Zone

current_generation = 0  # Generation counter


class Car:

    def __init__(self):
        # Load Car Sprite and Rotate
        # self.sprite = pygame.image.load('f1_car-removebg-preview.png').convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.image.load('car.png').convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        # self.position = [690, 740] # Starting Position
        self.position = [700, 820]  # Starting Position
        # self.position = [600, 875]  # Starting Position
        self.angle = 0
        self.speed = 0

        self.max_angular_acceleration = 0.1
        self.max_acceleration = 0.1

        self.stability_track = []

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]  # Calculate Center

        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.radars_out = []  # List For Outside Sensors / Radars
        self.radars_in = []  # List For Inside Sensors /  Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.alive = True  # Boolean To Check If Car is Crashed
        self.safe = True  # Boolean To Check If Car is in Safe Zone
        self.total_error_count = 0
        self.total_state_count = 0

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed
        self.safe_time = 0  # Increased if car is in safe zone, maybe decreased if outside of it not sure yet
        self.rewardModifier = {'green': 10, 'yellow': 1}

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        # self.draw_radar(screen)  # OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar_out in self.radars_out:
            position = radar_out[0]
            pygame.draw.line(screen, (200, 0, 0), self.center, position, 1)
            pygame.draw.circle(screen, (200, 0, 0), position, 5)

        for radar_in in self.radars_in:
            position = radar_in[0]
            pygame.draw.line(screen, (0, 0, 200), self.center, position, 1)
            pygame.draw.circle(screen, (0, 0, 200), position, 7)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if game_map.get_at((int(point[0]), int(point[1]))) == DEATH_ZONE:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length_out = 0
        length_in = 0
        x_out = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length_out)
        y_out = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length_out)

        x_in = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length_in)
        y_in = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length_in)

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while not game_map.get_at((x_out, y_out)) == DEATH_ZONE and length_out < 300:
            length_out = length_out + 1
            x_out = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length_out)
            y_out = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length_out)

        if self.is_safe(game_map):
            # game_map.get_at((x_in, y_in)) == SAFE_ZONE
            while not game_map.get_at((x_in, y_in)) == UNSAFE_ZONE and length_in < 250 and y_in < 1080 and y_in > 0 and x_in > 0 and x_in < 1920:
                length_in = length_in + 1
                x_in = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length_in)
                y_in = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length_in)

        # Calculate Distance To Border of Road And Append To Radars List
        dist_out = int(math.sqrt(math.pow(x_out - self.center[0], 2) + math.pow(y_out - self.center[1], 2)))
        dist_in = int(math.sqrt(math.pow(x_in - self.center[0], 2) + math.pow(y_in - self.center[1], 2)))
        self.radars_out.append([(x_out, y_out), dist_out])
        self.radars_in.append([(x_in, y_in), dist_in])

    def update(self, game_map):
        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 8
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        if self.is_safe(game_map):
            self.safe_time = self.safe_time+5
            self.total_state_count = self.total_state_count + 1
        else:
            self.safe_time = self.safe_time/1.2
            self.total_error_count = self.total_error_count + 1
            self.total_state_count = self.total_state_count + 1

        # Calculate New Center
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map)
        self.is_safe(game_map)
        self.radars_out.clear()
        self.radars_in.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 110, 30):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get Distances To Border
        if not self.radars_in:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0]

        return_values = [int(radar[1] / 30) for radar in self.radars_in]
        return_values.append(self.speed)
        return_values.append(0)

        return return_values

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def is_safe(self, game_map):
        left_top = self.corners[0]
        x_lt = int(left_top[0])
        y_lt = int(left_top[1])
        lt_col = game_map.get_at((x_lt, y_lt))

        right_top = self.corners[1]
        x_rt = int(right_top[0])
        y_rt = int(right_top[1])
        rt_col = game_map.get_at((x_rt, y_rt))

        left_bottom = self.corners[2]
        x_lb = int(left_bottom[0])
        y_lb = int(left_bottom[1])
        lb_col = game_map.get_at((x_lb, y_lb))

        right_bottom = self.corners[3]
        x_rb = int(right_bottom[0])
        y_rb = int(right_bottom[1])
        rb_col = game_map.get_at((x_rb, y_rb))

        if (lt_col == SAFE_ZONE) and (rt_col == SAFE_ZONE) and (lb_col == SAFE_ZONE) and (rb_col == SAFE_ZONE):
            self.safe = True
        else:
            self.safe = False
        return self.safe

    def how_safe(self, game_map):
        if self.is_safe(game_map):
            left_sensor = self.radars_in[0]
            right_sensor = self.radars_in[6]

            left_zone_width = left_sensor[1]
            right_zone_width = right_sensor[1]
            safe_zone_width = left_zone_width + right_zone_width

            if left_zone_width == right_zone_width:
                lane_ratio = 1
            elif left_zone_width > right_zone_width:
                delta = left_zone_width - safe_zone_width/2
                lane_ratio = float((safe_zone_width/2 - delta)/(safe_zone_width/2))
            elif right_zone_width > left_zone_width:
                delta = right_zone_width - safe_zone_width/2
                lane_ratio = float((safe_zone_width/2 - delta)/(safe_zone_width/2))
        else:
            lane_ratio = 0

        return lane_ratio

    def get_reward(self, game_map):
        # No reward if not moving
        if self.speed == 0:
            return 0

        # New Reward: Similar to Previous Except with Changed Safe Value and Bonus Value for Lane Ratio
        if self.safe:
            safe_value = self.rewardModifier['green']
        else:
            safe_value = self.rewardModifier['yellow']

        bonus_safe_value = 10
        lane_ratio = self.how_safe(game_map)

        reward = (safe_value + bonus_safe_value*lane_ratio + (self.speed - 6))/20 + self.distance/100

        return reward

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

def run_training(genomes, config):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []

    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    # game_map = pygame.image.load('testing_track3.png').convert()  # Convert Speeds Up A Lot
    global tmap

    # random number between 0 and len(maps)
    # tmap = np.random.choice(maps)
    game_map = pygame.image.load(tmap).convert()  # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0

    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Get The Acton It Takes

        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())

            delta_angle = (output[0] - 0.5) * 2 * 7
            delta_speed = (output[1] - 0.5) * 2 * 0.1

            # Adjust the turning angle based on speed to simulate skidding
            # The faster the car is, the less effective the steering should be
            # If the car is in the safe zone, the skid factor is reduced
            max_reduction = 0.4 if car.safe else 0.8
            skid_factor = 1 - np.clip((car.speed - 6) / 6, 0, 1) * max_reduction
            adjusted_delta_angle = delta_angle * skid_factor

            car.angle += adjusted_delta_angle
            car.speed += delta_speed

            # Bound speed between 4 and 12
            car.speed = np.clip(car.speed, 4, 12)


        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward(game_map)

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:  # Stop After About 20 Seconds
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = generation_font.render("Generation: " + str(current_generation), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 630)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 670)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)  # 60 FPS

    global survPerGen
    global gen_i
    survPerGen[gen_i] = still_alive


    # Select best genome
    max_fitness = 0
    max_cars = []
    max_genomes = []
    max_d = 0
    global distPerGen
    for i, car in enumerate(cars):
        if car.distance > max_d:
            bestDistPerGen[gen_i] = car.distance
            max_d = car.distance
        if car.is_alive():
            if genomes[i][1].fitness == max_fitness:
                max_cars.append(car)
                max_genomes.append(genomes[i])
            elif genomes[i][1].fitness > max_fitness:
                max_fitness = genomes[i][1].fitness
                max_cars = [car]
                max_genomes = [genomes[i]]
    gen_i = gen_i +1


    if not max_genomes:
        print("All Cars Crashed")
    else:
        # Save winner
        # genome_path = "Winner.pkl"
        winner = max_genomes[0]
        with open(genome_path, "wb") as f:
            pickle.dump(winner, f)
            f.close()

def run_testing(genomes, config, map):
    # Empty Collections For Nets and Cars
    nets = []
    cars = []
    track = map


    # Initialize PyGame And The Display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    # For All Genomes Passed Create A New Neural Network
    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        cars.append(Car())

    # Clock Settings
    # Font Settings & Loading Map
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load(track).convert()  # Convert Speeds Up A Lot

    global current_generation
    current_generation += 1

    speedPerTimeStep = np.zeros([1, 1200])[0]
    distPerTimeStep = np.zeros([1, 1200])[0]
    safetyRatPerTimeStep = np.zeros([1, 1200])[0]

    # Simple Counter To Roughly Limit Time (Not Good Practice)
    counter = 0
    t=0
    while True:
        # Exit On Quit Event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        # For Each Car Get The Acton It Takes
        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())

            delta_angle = (output[0] - 0.5) * 2 * 7
            delta_speed = (output[1] - 0.5) * 2 * 0.1

            # Adjust the turning angle based on speed to simulate skidding
            # The faster the car is, the less effective the steering should be
            # If the car is in the safe zone, the skid factor is reduced
            max_reduction = 0.4 if car.safe else 0.8
            skid_factor = 1 - np.clip((car.speed - 6) / 6, 0, 1) * max_reduction
            adjusted_delta_angle = delta_angle * skid_factor

            car.angle += adjusted_delta_angle
            car.speed += delta_speed

            # Bound speed between 4 and 12
            car.speed = np.clip(car.speed, 4, 12)

        speedPerTimeStep[t] = car.speed
        distPerTimeStep[t] = car.distance
        t=t+1
        # Check If Car Is Still Alive
        # Increase Fitness If Yes And Break Loop If Not
        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                safetyRatPerTimeStep[t-1] = car.how_safe(game_map)
                genomes[i][1].fitness += car.get_reward(game_map)

        if still_alive == 0:
            print("Car Crashed on Track " + str(track))
            break

        counter += 1
        if counter == 30 * 40:  # Stop After About 20 Seconds
            error_count = car.total_error_count
            total_count = car.total_state_count
            safe_count = total_count - error_count
            safe_ratio = safe_count / total_count
            st_sr = "{:.3f}".format(safe_ratio)
            # print("Success! Track " + str(track) + " Safety Ratio: " + str(st_sr))
            break

        # Draw Map And All Cars That Are Alive
        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Display Info
        text = generation_font.render("Best Genome from EANN on Testing Track " + str(track), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 950)
        screen.blit(text, text_rect)

        error_count = car.total_error_count
        total_count = car.total_state_count
        safe_count = total_count - error_count
        safe_ratio = safe_count / total_count
        distance = car.distance
        speed = car.speed

        st_d = "{:.3f}".format(distance)
        st_sr = "{:.3f}".format(safe_ratio)
        st_sp = "{:.3f}".format(speed)

        text = generation_font.render("Safety Ratio " + str(safe_count) + "/" + str(total_count) + " = " + str(st_sr), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 1000)
        screen.blit(text, text_rect)

        text = generation_font.render("Distance = " + str(st_d), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (800, 1050)
        screen.blit(text, text_rect)

        text = generation_font.render("Speed = " + str(st_sp), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (1030, 1050)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(50)  # 60 FPS

    survived = cars[0].is_alive()
    percent_on_green = st_sr
    # dist_covered = total_count
    dist_covered = cars[0].distance
    fitness_value = genomes[0][1].fitness
    return survived, still_alive, percent_on_green, dist_covered, fitness_value, speedPerTimeStep, distPerTimeStep, safetyRatPerTimeStep


def train_net(config_path, genome_path, gen_count):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    # Create Population And Add Reporters
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    global survPerGen
    survPerGen = np.zeros([1, gen_count])[0]
    global bestDistPerGen
    bestDistPerGen = np.zeros([1, gen_count])[0]
    global gen_i
    gen_i = 0
    population.run(run_training, gen_count)
    # print(s_count)


if __name__ == "__main__":
    # Load Config
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    maps = ['testing_track1.png', 'testing_track2.png', 'testing_track3.png', 'testing_track4.png', 'testing_track5.png']
    # maps = ['testing_track1.png', 'testing_track2.png']
    global genome_path
    global tmap

    #Setup Variables
    gen_count = 3
    n = 5
    train_map_idx = 3
    eval_map_idx = 1
    tmap = maps[train_map_idx]
    genome_path = "Winner2.pkl"

    potentialWhatToRunCmds = ['all', '1train4test', 'survivorgraph', 'DistancesAndGreenRatio', 'resultsTrainedBy1']
    whattorun = ['trainingGraphs', 'crossVerificationGraphs']

    gs = [gen_count]

    #Data collection Data Structures
    distances = np.zeros((len(maps), len(gs)))
    greenrats = np.zeros((len(maps), len(gs)))
    survivors = np.zeros((len(maps), max(gs)))
    mapdistPerGen = np.zeros((len(maps), max(gs)))


    # Generate Graphs for Training
    # Collect Survivors per generation and distance covered per generation
    if ('all' in whattorun) or ('trainingGraphs' in whattorun):
        tgen_count = max(gs)
        for _ in range(0,n):
            for ij in range(0, len(maps)):
                tmap = maps[ij]
                train_net(config_path, genome_path, tgen_count)
                survivors[ij] = survivors[ij] + copy.copy(survPerGen)
                mapdistPerGen[ij] = mapdistPerGen[ij] + copy.copy(bestDistPerGen)
                print(survPerGen)
                print(bestDistPerGen)
                print()
        survivors = [[x / n for x in subl] for subl in survivors]
        mapdistPerGen = [[x / n for x in subl] for subl in mapdistPerGen]


        for k in range(0, len(maps)):
            plt.plot(survivors[k], label=maps[k].split('.')[0])
        plt.xlabel('Num of Generations')
        plt.ylabel('Survivors')
        plt.title('Survivors per Generation')
        plt.legend(loc='upper right')
        plt.savefig('TrainingSurvivors.png')
        plt.show()

        for k in range(0, len(maps)):
            plt.plot(mapdistPerGen[k], label=maps[k].split('.')[0])
        plt.xlabel('Num of Generations')
        plt.ylabel('Max Distance Covered')
        plt.title('Distance Covered per Generation')
        plt.legend(loc='upper right')
        plt.savefig('TrainingDistance.png')
        plt.show()


    # Generate Cross Verifiaction Table
    # Collect survivors after training on each track
    if ('all' in whattorun) or ('crossVerificationGraphs' in whattorun):
        trainedTable = np.zeros((len(maps), len(maps)))
        CrossVerificationTable = np.zeros((len(maps), len(maps)))
        HiddenNodesCount = np.zeros((len(maps), len(maps)))
        GreenRatioTable = np.zeros((len(maps), len(maps)))

        for ii in range(0, n):
            print("Iterate through tables: " + str(ii) + " out of " + str(n))
            print(CrossVerificationTable)
            for c in range(0, len(maps)):
                tmap = maps[c]

                train_net(config_path, genome_path, gen_count)

                if os.path.exists(genome_path):
                    trainedTable[c][ii] = 1
                    pass
                else:
                    continue

                # Evaluate performance on other maps
                with open(genome_path, "rb") as f:
                    genomes = [pickle.load(f)]

                cs = genomes[0][1].connections
                inp_count = 0
                for i in cs:
                    if i[0] < 0:
                        if abs(i[0]) > inp_count:
                            inp_count = abs(i[0])
                num_nodes = len(genomes[0][1].nodes.keys())
                hidden_nodes_count = num_nodes - inp_count
                HiddenNodesCount[c][ii] = hidden_nodes_count

                for j in range(0, len(maps)):
                    config = neat.config.Config(neat.DefaultGenome,
                                                neat.DefaultReproduction,
                                                neat.DefaultSpeciesSet,
                                                neat.DefaultStagnation,
                                                config_path)
                    survived, still_alive, percent_on_green, dist_covered, fitness_value, speedPerTimeStep, distPerTimeStep, safetyRatPerTimeStep = run_testing(genomes,
                                                                                                       config,
                                                                                                       maps[j])
                    if survived:
                        CrossVerificationTable[c][j] += 1
                    GreenRatioTable[c][j] = GreenRatioTable[c][j] + float(percent_on_green)

                if os.path.exists(genome_path):
                    os.remove(genome_path)

        print("Success Count Table:")
        print(CrossVerificationTable)
        r = np.array(CrossVerificationTable) / n
        print("Success Percentage Table:")
        print(r)
        print("Training Success Table:")
        print(trainedTable)
        print("Hidden Node Final Count Table:")
        print(HiddenNodesCount)
        print("Green Ratio Table:")
        print(GreenRatioTable)

        file1 = open("trainedTable.txt", "w")
        str1 = repr(trainedTable)
        file1.write(str1)

        file2 = open("SuccessPercentageTable.txt", "w")
        str2 = repr(r)
        file2.write(str2)

        file3 = open("GreenRatioTable.txt", "w")
        str3 = repr(GreenRatioTable)
        file3.write(str2)


    # Deep Dive
    # Train on 1
    # Collect speeds per time
    # Collect safety ratio
    # Collect distances
    #TODO: Add code for capturing safty ratio
    if ('all' in whattorun) or ('deepDiveGraphs' in whattorun):
        # genome_path = "Winner3.pkl"
        deepSpeed = np.zeros([len(maps), 1200])
        deepLaneSafety = np.zeros([len(maps), 1200])
        deepDistances = np.zeros((len(maps), 1200))
        deepGreen = np.zeros((len(maps), max(gs)))


        train_net(config_path, genome_path, gen_count)

        if os.path.exists(genome_path):
            pass
        else:
            print("Eval Complete")
            sys.exit()


        with open(genome_path, "rb") as f:
            genomes = [pickle.load(f)]

        # Evaluate performance on other maps
        for j in range(0, len(maps)):
            config = neat.config.Config(neat.DefaultGenome,
                                        neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet,
                                        neat.DefaultStagnation,
                                        config_path)
            survived, still_alive, percent_on_green, dist_covered, fitness_value, speedPerTimeStep, distPerTimeStep, safetyRatPerTimeStep = run_testing(genomes, config,
                                                                                               maps[j])
            deepSpeed[j] = speedPerTimeStep
            deepDistances[j] = distPerTimeStep
            deepLaneSafety[j] = safetyRatPerTimeStep

            print(survived, dist_covered, percent_on_green)

            # print(speedPerTimeStep)
            # print(survived, still_alive, percent_on_green, dist_covered, fitness_value)
            # sys.exit()


        # for k in range(0, len(maps)):
        k = eval_map_idx
        plt.plot(deepSpeed[k], label=maps[k].split('.')[0])
        plt.xlabel('Time Step')
        plt.ylabel('Speed')
        plt.title('Solution Speed over Time')
        plt.legend(loc='upper right')
        plt.savefig('deepSpeed.png')
        plt.show()

        for k in range(0, len(maps)):
            plt.plot(deepDistances[k], label=maps[k].split('.')[0])
        plt.xlabel('Time Step')
        plt.ylabel('Distance Covered')
        plt.title('Distance Covered Over Time')
        plt.legend(loc='upper right')
        plt.savefig('deepDistances.png')
        plt.show()

        # for k in range(0, len(maps)):
        k = eval_map_idx
        plt.plot(deepLaneSafety[k], label=maps[k].split('.')[0])
        plt.xlabel('Time Step')
        plt.ylabel('Safety Ratio')
        plt.title('Lane Safety Ratio Over Time')
        plt.legend(loc='upper right')
        plt.savefig('deepLaneSafety.png')
        plt.show()



    print("Eval finished")
