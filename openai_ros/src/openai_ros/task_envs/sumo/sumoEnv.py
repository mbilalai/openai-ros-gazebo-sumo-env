import rospy
import numpy
from gym import spaces
from openai_ros.robot_envs import turtlebot2sumo_env
from gym.envs.registration import register
from geometry_msgs.msg import Point
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
from openai_ros.openai_ros_common import ROSLauncher
import os


class SumoEnv(turtlebot2sumo_env.TurtleBot2SumoEnv):
    def __init__(self):
        """
        This Task Env is designed for having the TurtleBot2 in some kind of maze.
        It will learn how to move around the maze without crashing.
        """

        # This is the path where the simulation files, the Task and the Robot gits will be downloaded if not there
        ros_ws_abspath = rospy.get_param("/turtlebot2/ros_ws_abspath", None)
        assert ros_ws_abspath is not None, "You forgot to set ros_ws_abspath in your yaml file of your main RL script. Set ros_ws_abspath: \'YOUR/SIM_WS/PATH\'"
        assert os.path.exists(ros_ws_abspath), "The Simulation ROS Workspace path " + ros_ws_abspath + \
                                               " DOESNT exist, execute: mkdir -p " + ros_ws_abspath + \
                                               "/src;cd " + ros_ws_abspath + ";catkin_make"

        ROSLauncher(rospackage_name="turtlebot_gazebo",
                    launch_file_name="sumo_world.launch",
                    ros_ws_abspath=ros_ws_abspath)

        rospy.logdebug("finish loading sumo_world.launch")

        # Load Params from the desired Yaml file
        LoadYamlFileParamsTest(rospackage_name="openai_ros",
                               rel_path_from_package_to_file="src/openai_ros/task_envs/sumo/config",
                               yaml_file_name="sumo.yaml")

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(SumoEnv, self).__init__(ros_ws_abspath)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot2/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        
        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
        
        
        #number_observations = rospy.get_param('/turtlebot2/n_observations')
        """
        We set the Observation space for the 6 observations
        cube_observations = [
            round(current_disk_roll_vel, 0),
            round(y_distance, 1),
            round(roll, 1),
            round(pitch, 1),
            round(y_linear_speed,1),
            round(yaw, 1),
        ]
        """
        
        # Actions and Observations
        self.linear_forward_speed = rospy.get_param('/turtlebot2/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot2/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot2/angular_speed')
        self.init_linear_forward_speed = rospy.get_param('/turtlebot2/init_linear_forward_speed')
        self.init_linear_turn_speed = rospy.get_param('/turtlebot2/init_linear_turn_speed')
        #self.init_linear_forward_speed = numpy.random.uniform(-1,1)
        #self.init_linear_turn_speed = numpy.random.uniform(-1,1)
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        #laser_scan = self.get_laser_scan()
        #rospy.logdebug("laser_scan len===>" + str(len(laser_scan.ranges)))

        
        # We only use two integers
        self.observation_space = spaces.Box(low=0, high=255, shape= (640, 480, 3))
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))
        
        # Rewards
        self.hit_reward = rospy.get_param("/turtlebot2/hit_reward")
        self.robot_out_of_bounds_penalty = rospy.get_param("/turtlebot2/robot_out_of_bounds_penalty")
        self.ball_out_of_bounds_reward = rospy.get_param("/turtlebot2/ball_out_of_bounds_reward")

        self.cumulated_steps = 0.0
        



    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base( self.init_linear_forward_speed,
                        self.init_linear_turn_speed,
                        sleep_time=0,
                        epsilon=0.05,
                        update_rate=10)

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        
        camera_data = self.get_camera_rgb_image_raw()


    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        if action == 0: #FORWARD
            linear_speed = self.linear_forward_speed
            angular_speed = 0.0
            self.last_action = "FORWARDS"
        elif action == 1: #LEFT
            linear_speed = self.linear_turn_speed
            angular_speed = self.angular_speed
            self.last_action = "TURN_LEFT"
        elif action == 2: #RIGHT
            linear_speed = self.linear_turn_speed
            angular_speed = -1*self.angular_speed
            self.last_action = "TURN_RIGHT"

        
        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(linear_speed, angular_speed, epsilon=0.05, update_rate=10)
        
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        observations = self.get_camera_rgb_image_raw()
        rospy.logdebug("END Get Observation ==>")
        return observations
        

    def _is_done(self, observations):
        
        if self._episode_done:
            rospy.logerr("TurtleBot2 is Too Close to wall==>")
        else:
            current_position = self.get_robot_position()
            ball_position = self.get_ball_position()
            #rospy.logerr("TurtleBot2 didnt crash at least ==>")
            
            
            MAX_X = 2.5
            MIN_X = -2.5
            MAX_Y = 2.5
            MIN_Y = -2.5
            
            # We see if we are outside the Learning Space
            
            if current_position[0] > MAX_X or current_position[0] < MIN_X or current_position[1] > MAX_Y or current_position[1] < MIN_Y:
                rospy.logerr("TurtleBot to Far in Y Pos ==>"+str(current_position[0]))
                self._episode_done = True
                self.win = -1
            
            if ball_position[0] > MAX_X or ball_position[0] < MIN_X or ball_position[1] > MAX_Y or ball_position[1] < MIN_Y:
                rospy.logerr("TurtleBot to Far in Y Pos ==>"+str(ball_position[0]))
                self._episode_done = True
                self.win = 1            

        return self._episode_done

    def _compute_reward(self, observations, done):
        reward = 0.0
        current_position = numpy.array(self.get_robot_position())
        ball_position = numpy.array(self.get_ball_position())
        r_robot = 0.351/2
        r_ball = 0.5
        p = 0.1

        if not done:
            if (r_robot + r_ball + p) <= numpy.linalg.norm(current_position - ball_position):
                reward += self.hit_reward
                
        else:            
            if self.win == 1:
                reward = self.ball_out_of_bounds_reward/(self.step_number+1.0)
            else:
                reward = self.robot_out_of_bounds_penalty




        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward


    # Internal TaskEnv Methods
