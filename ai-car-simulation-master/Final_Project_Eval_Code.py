# This Code is Heavily Inspired By The YouTuber: Cheesy AI
# Code Changed, Optimized And Commented By: NeuralNine (Florian Dedov)

import math
import random
import sys
import os

import neat
import pygame

# Constants
# WIDTH = 1600
# HEIGHT = 880

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 40
CAR_SIZE_Y = 40

DEATH_ZONE = (255, 255, 255, 255)  # Color To Crash on Hit
SAFE_ZONE = (200, 200, 0, 255)
UNSAFE_ZONE = (255, 255, 16, 255)

current_generation = 0  # Generation counter


class Car:

    def __init__(self):
        # Load Car Sprite and Rotate
        self.sprite = pygame.image.load('f1_car-removebg-preview.png').convert()  # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        # self.position = [690, 740] # Starting Position
        self.position = [700, 840]  # Starting Position
        # self.position = [600, 875]  # Starting Position
        self.angle = 0
        self.speed = 0

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

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed
        self.safe_time = 0  # Increased if car is in safe zone, maybe decreased if outside of it not sure yet


    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        self.draw_radar(screen)  # OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar_out in self.radars_out:
            position = radar_out[0]
            pygame.draw.line(screen, (200, 0, 0), self.center, position, 1)
            pygame.draw.circle(screen, (200, 0, 0), position, 5)

        for radar_in in self.radars_in:
            position = radar_in[0]
            pygame.draw.line(screen, (0, 0, 200), self.center, position, 1)
            pygame.draw.circle(screen, (0, 0, 200), position, 5)

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
            self.speed = 20
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        if self.is_safe(game_map):
            self.safe_time = self.safe_time+5
        else:
            self.safe_time = self.safe_time/1.2

        # Same For Y-Position
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

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
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        # Get Distances To Border
        radars_out = self.radars_in
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars_out):
            return_values[i] = int(radar[1] / 30)  # radar[1] is the distance to the pixel that the sensor is sensing
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
            right_sensor = self.radars_in[4]

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
        # Calculate Reward (Maybe Change?)
        # return self.distance / 50.0
        total_safe_time = self.safe_time
        lane_ratio = self.how_safe(game_map)
        return (total_safe_time * lane_ratio) / (CAR_SIZE_X / 2) + (self.distance/10)/(CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image


def run_simulation(genomes, config):
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
    game_map = pygame.image.load('curvy_track.png').convert()  # Convert Speeds Up A Lot

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
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 4  # Left
            elif choice == 1:
                car.angle -= 4  # Right
            elif choice == 2:
                if (car.speed - 2 >= 12):
                    car.speed -= 2  # Slow Down
            elif choice == 3:
                car.speed += 2  # Speed Up
            elif choice == 4:  # Maintain Speed and Heading
                car.speed = car.speed + 0
                car.angle = car.angle + 0

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


if __name__ == "__main__":
    # Load Config
    config_path = "./config.txt"
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

    # Run Simulation For A Maximum of 1000 Generations
    population.run(run_simulation, 30)
